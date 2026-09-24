"""SANED classification, LDS regression, and cross-subject InfoNCE losses."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

def lds_sample_weights(values: np.ndarray, bins: int = 50, sigma: float = 0.05) -> np.ndarray:
    """为回归目标估计标签分布平滑权重。"""
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, -1.0, 1.0)
    edges = np.linspace(-1.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(clipped, edges) - 1, 0, bins - 1)
    hist = np.bincount(idx, minlength=bins).astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    kernel = np.exp(-0.5 * ((centers[:, None] - centers[None, :]) / max(sigma, 1e-6)) ** 2)
    kernel /= np.maximum(kernel.sum(axis=1, keepdims=True), 1e-12)
    effective = hist @ kernel
    weights = 1.0 / np.maximum(effective[idx], 1e-6)
    weights /= np.mean(weights)
    return weights.astype(np.float32)


class LDSWeightLookup:
    """将目标值映射为训练折拟合得到的固定 LDS 权重。"""

    def __init__(self, values: np.ndarray, bins: int = 50, sigma: float = 0.05):
        """根据训练目标值拟合平滑的逆频率权重。"""
        self.edges = np.linspace(-1.0, 1.0, bins + 1)
        self.weights = lds_sample_weights(values, bins=bins, sigma=sigma)
        clipped = np.clip(values, -1.0, 1.0)
        idx = np.clip(np.digitize(clipped, self.edges) - 1, 0, bins - 1)
        self.bin_weights = np.zeros(bins, dtype=np.float32)
        for i in range(bins):
            vals = self.weights[idx == i]
            self.bin_weights[i] = float(vals.mean()) if len(vals) else 1.0

    def tensor(self, values: torch.Tensor) -> torch.Tensor:
        """为目标值张量查找已经拟合好的分箱权重。"""
        edges = torch.as_tensor(self.edges, device=values.device, dtype=values.dtype)
        # 与拟合直方图时使用的 np.digitize(..., right=False) 保持一致：
        # 恰好落在内部边界的值归入右侧分箱。
        bins = torch.bucketize(values.clamp(-1.0, 1.0), edges[1:-1], right=True)
        weights = torch.as_tensor(self.bin_weights, device=values.device, dtype=values.dtype)
        return weights[bins]


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """计算逐样本加权的均方误差。"""
    return (weights * (pred - target).square()).mean()


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """计算忽略非分类样本、的交叉熵。"""
    valid = target != -100
    if not torch.any(valid):
        # 当打乱后的批次没有分类样本时，避免 reduction='mean' 产生 NaN。
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(
        logits, target, ignore_index=-100
    )


def multi_positive_infonce(
    embeddings: torch.Tensor,
    subjects: list[str] | tuple[str, ...],
    events: torch.Tensor | list[int],
    temperature: float = 0.10,
) -> torch.Tensor:
    """计算同事件、不同被试的多正样本 InfoNCE。

    一个 anchor 的正样本是 batch 内 ``event`` 相同且 ``subject`` 不同的样本；
    其余非自身样本作为分母候选。没有正样本的 anchor 被跳过，允许最后一个
    不完整 batch  正常运行。
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be [batch, dim], got {tuple(embeddings.shape)}")
    batch = embeddings.shape[0]
    if batch < 2:
        return embeddings.sum() * 0.0
    if len(subjects) != batch:
        raise ValueError("subjects length must match embeddings batch")
    if isinstance(events, torch.Tensor):
        events_tensor = events.to(device=embeddings.device)
    else:
        event_values = list(events)
        event_to_id = {value: i for i, value in enumerate(dict.fromkeys(event_values))}
        events_tensor = torch.as_tensor(
            [event_to_id[value] for value in event_values], device=embeddings.device
        )
    if events_tensor.numel() != batch:
        raise ValueError("events length must match embeddings batch")
    normalized = nn.functional.normalize(embeddings, dim=-1)
    logits = torch.matmul(normalized, normalized.transpose(0, 1)) / temperature
    eye = torch.eye(batch, dtype=torch.bool, device=embeddings.device)
    same_event = events_tensor[:, None].eq(events_tensor[None, :])
    same_subject = torch.tensor(
        [[a == b for b in subjects] for a in subjects],
        dtype=torch.bool,
        device=embeddings.device,
    )
    # 同事件且同被试的重复窗口不是负样本；负样本必须来自不同事件。
    valid_denominator = ~eye & ~(same_event & same_subject)
    logits = logits.masked_fill(~valid_denominator, -torch.inf)
    positives = same_event & ~same_subject & valid_denominator
    has_positive = positives.any(dim=1)
    if not torch.any(has_positive):
        return embeddings.sum() * 0.0
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positives.sum(dim=1).clamp_min(1)
    per_anchor = -(log_prob.masked_fill(~positives, 0.0).sum(dim=1) / positive_count)
    return per_anchor[has_positive].mean()

