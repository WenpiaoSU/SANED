"""SANED source-space emotion decoding with three-sentence context."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..config import Config
from .spatial import ROISpatialEncoder
from .temporal import MultiScaleROITemporalEncoder


class ContextROIGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(tokens.mean(dim=2)), dim=1).unsqueeze(2)
        return tokens * (1.0 + weights)


class ContextConvBlock(nn.Module):
    """Residual convolution along sentences, preserving aligned ROI identities."""
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.temporal = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        update = self.temporal(tokens.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.norm(tokens + self.dropout(nn.functional.gelu(update)))


class SANED(nn.Module):
    def __init__(self, cfg: Config, token_positions: np.ndarray, token_hemispheres: np.ndarray):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.temporal_encoder = MultiScaleROITemporalEncoder(
            cfg.n_bands, cfg.dim, cfg.temporal_kernel_sizes, cfg.temporal_dilations, cfg.dropout
        )
        self.spatial_encoder = ROISpatialEncoder(cfg, token_positions, token_hemispheres)
        self.context_gate = ContextROIGate(cfg.dim)
        self.context_blocks = nn.ModuleList(
            [ContextConvBlock(cfg.dim, cfg.dropout) for _ in range(cfg.context_blocks)]
        )
        self.embedding_head = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, cfg.contrastive_dim))
        self.emotion_head = nn.Sequential(
            nn.Dropout(cfg.dropout), nn.Linear(cfg.contrastive_dim, cfg.hidden_dim),
            nn.GELU(), nn.Dropout(cfg.dropout),
        )
        self.classifier = nn.Linear(cfg.hidden_dim, cfg.n_classes)
        self.valence_head = nn.Linear(cfg.hidden_dim, 1)
        self.arousal_head = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        expected = (3, self.cfg.n_rois, self.cfg.n_bands, self.cfg.target_length)
        if x.ndim != 5 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"SANED expects [batch, {', '.join(map(str, expected))}], got {tuple(x.shape)}")
        batch, context, rois, bands, time = x.shape
        encoded = self.temporal_encoder(x.reshape(batch * context * rois, bands, time))
        tokens = self.spatial_encoder.encode_tokens(encoded.reshape(batch, context, rois, -1))
        tokens = self.context_gate(tokens)
        for block in self.context_blocks:
            tokens = block(tokens)
        sentence_tokens, roi_weights = self.spatial_encoder.pool_encoded_tokens(tokens)
        embedding = self.embedding_head(sentence_tokens.mean(dim=1))
        hidden = self.emotion_head(embedding)
        return {
            "class_logits": self.classifier(hidden),
            "valence": self.valence_head(hidden).squeeze(-1),
            "arousal": self.arousal_head(hidden).squeeze(-1),
            "embedding": embedding,
            "roi_tokens": tokens.mean(dim=1),
            "roi_weights": roi_weights.mean(dim=1),
        }
