"""Shared deterministic preprocessing helpers for SANED."""

from __future__ import annotations

import numpy as np


def resample_segment(segment: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample a ``[features, time]`` segment to a fixed length."""
    n = int(segment.shape[-1])
    if n < 1:
        raise ValueError("Cannot resample an empty segment")
    if n < 2:
        return np.repeat(segment, target_length, axis=-1).astype(np.float32, copy=False)
    positions = np.linspace(0.0, n - 1.0, target_length, dtype=np.float64)
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, n - 1)
    weight = (positions - left).astype(np.float32)
    return (
        segment[:, left] * (1.0 - weight)[None, :]
        + segment[:, right] * weight[None, :]
    ).astype(np.float32, copy=False)
