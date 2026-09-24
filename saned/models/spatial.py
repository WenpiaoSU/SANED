"""Anatomical embeddings, ROI Transformer, and attention pooling."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ROISpatialEncoder(nn.Module):
    def __init__(self, cfg, token_positions: np.ndarray, token_hemispheres: np.ndarray):
        super().__init__()
        self.n_tokens, self.dim = cfg.n_rois, cfg.dim
        positions = np.asarray(token_positions, dtype=np.float32)
        hemispheres = np.asarray(token_hemispheres, dtype=np.int64)
        if positions.shape != (cfg.n_rois, 3) or not np.isfinite(positions).all():
            raise ValueError(f"Expected finite ROI coordinates [{cfg.n_rois}, 3]")
        if hemispheres.shape != (cfg.n_rois,) or not np.isin(hemispheres, [0, 1]).all():
            raise ValueError(f"Expected binary hemisphere labels [{cfg.n_rois}]")
        self.roi_identity = nn.Parameter(torch.empty(1, cfg.n_rois, cfg.dim))
        self.roi_position = nn.Parameter(torch.empty(1, cfg.n_rois, cfg.dim))
        nn.init.normal_(self.roi_identity, std=0.02)
        nn.init.normal_(self.roi_position, std=0.02)
        scale = np.maximum(np.std(positions, axis=0, keepdims=True), 1e-6)
        positions = (positions - positions.mean(axis=0, keepdims=True)) / scale
        self.register_buffer("token_positions", torch.from_numpy(positions), persistent=False)
        self.register_buffer("token_hemispheres", torch.from_numpy(hemispheres), persistent=False)
        self.position_projection = nn.Sequential(nn.Linear(3, cfg.dim), nn.LayerNorm(cfg.dim), nn.GELU())
        self.hemisphere_embedding = nn.Embedding(2, cfg.dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.dim, nhead=cfg.heads, dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.layers, norm=nn.LayerNorm(cfg.dim))
        self.roi_score = nn.Linear(cfg.dim, 1)

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, context, n_tokens, dim = tokens.shape
        tokens = tokens + self.roi_identity.unsqueeze(1) + self.roi_position.unsqueeze(1)
        tokens = tokens + self.position_projection(self.token_positions).view(1, 1, n_tokens, dim)
        tokens = tokens + self.hemisphere_embedding(self.token_hemispheres).view(1, 1, n_tokens, dim)
        return self.encoder(tokens.reshape(batch * context, n_tokens, dim)).reshape(
            batch, context, n_tokens, dim
        )

    def pool_encoded_tokens(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.roi_score(encoded).squeeze(-1), dim=-1)
        return (encoded * weights.unsqueeze(-1)).sum(dim=2), weights
