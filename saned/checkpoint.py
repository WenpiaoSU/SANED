"""Self-contained SANED checkpoints and strict weight loading."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .models import SANED


def save_json(value, path: Path) -> None:
    """Represent undefined metrics with JSON null."""
    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(val) for val in item]
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, np.integer):
            return int(item)
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_weights(model: SANED, state: dict) -> None:
    state = dict(state)
    # Some weight exports include this all-ROI bookkeeping buffer.
    mask = state.pop("spatial_encoder.configured_mask", None)
    if mask is not None and (tuple(mask.shape) != (model.cfg.n_rois,) or not bool(mask.all())):
        raise ValueError("The checkpoint must contain the complete ROI set")
    model.load_state_dict(state, strict=True)


def make_checkpoint(model, cfg, mean, std, metadata, split, epoch, score) -> dict:
    positions, hemispheres, names = metadata
    return {
        "model_name": "SANED",
        "config": cfg.to_dict(),
        "model": {key: val.detach().cpu().clone() for key, val in model.state_dict().items()},
        "normalization_mean": torch.as_tensor(mean, dtype=torch.float32),
        "normalization_std": torch.as_tensor(std, dtype=torch.float32),
        "roi_positions": torch.as_tensor(positions, dtype=torch.float32),
        "roi_hemispheres": torch.as_tensor(hemispheres, dtype=torch.long),
        "roi_names": names,
        "split": split,
        "epoch": epoch,
        "validation_score": score,
        "selection_metric": "valence_mse+arousal_mse",
    }


def load_checkpoint(path: str | Path, device: str = "cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"config", "model", "normalization_mean", "normalization_std",
                "roi_positions", "roi_hemispheres", "roi_names", "split"}
    if not isinstance(checkpoint, dict) or required - set(checkpoint):
        raise ValueError("Expected a complete SANED checkpoint; use saned.export_checkpoint for raw weights")
    if checkpoint.get("model_name") != "SANED":
        raise ValueError("Expected a SANED checkpoint")
    cfg = Config(**checkpoint["config"])
    cfg.validate()
    mean = checkpoint["normalization_mean"].numpy()
    std = checkpoint["normalization_std"].numpy()
    if mean.shape != (cfg.n_rois, cfg.n_bands) or std.shape != mean.shape:
        raise ValueError("Invalid normalization shape")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid normalization values")
    model = SANED(cfg, checkpoint["roi_positions"].numpy(), checkpoint["roi_hemispheres"].numpy())
    load_weights(model, checkpoint["model"])
    model.to(device).eval()
    return model, cfg, checkpoint, mean, std
