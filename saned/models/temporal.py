"""Multi-scale ROI temporal encoding for SANED."""
from __future__ import annotations
import torch
from torch import nn

class TemporalAttentionPool(nn.Module):
    """对卷积序列做可学习的时间注意力池化。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 ``[items, channels, time]`` 映射为 ``[items, channels]``。"""
        weights = torch.softmax(self.score(x).squeeze(1), dim=-1)
        return (x * weights.unsqueeze(1)).sum(dim=-1)


class MultiScaleROITemporalEncoder(nn.Module):
    """用短、中、长三个时间尺度编码一个 ROI 的频段序列。"""

    def __init__(
        self,
        n_bands: int,
        dim: int,
        kernel_sizes: tuple[int, int, int],
        dilations: tuple[int, int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        if len(kernel_sizes) != 3 or len(dilations) != 3:
            raise ValueError("SANED temporal encoder requires exactly three branches")
        if dim < 24:
            raise ValueError("SANED temporal encoder dimension must be at least 24")
        self.band_gate = nn.Sequential(
            nn.Linear(n_bands, n_bands), nn.GELU(), nn.Linear(n_bands, n_bands)
        )
        branch_dim = max(8, dim // 3)
        branch_dims = (branch_dim, branch_dim, dim - 2 * branch_dim)
        branches: list[nn.Module] = []
        for channels, kernel, dilation in zip(branch_dims, kernel_sizes, dilations):
            if kernel < 1 or kernel % 2 == 0:
                raise ValueError("SANED temporal kernels must be positive odd integers")
            if dilation < 1:
                raise ValueError("SANED temporal dilations must be positive")
            padding = dilation * (kernel - 1) // 2
            branches.append(
                nn.Sequential(
                    nn.Conv1d(n_bands, channels, kernel, dilation=dilation, padding=padding),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),
                    nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=padding),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                    TemporalAttentionPool(channels),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.projection = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 ``[items, bands, time]`` 编码为 ``[items, dim]``。"""
        if x.ndim != 3:
            raise ValueError(f"Expected [items, bands, time], got {tuple(x.shape)}")
        band_weights = torch.softmax(self.band_gate(x.mean(dim=-1)), dim=-1)
        gated = x * band_weights.unsqueeze(-1)
        features = [branch(gated) for branch in self.branches]
        return self.projection(torch.cat(features, dim=-1))

