"""Final SANED architecture and training defaults."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CACHE_ROOT = PROJECT_ROOT / "data" / "source_cache"
ANNOTATION_PATH = SOURCE_CACHE_ROOT / "sentence_manifest.tsv"
MEG_SEGMENT_ROOT = PROJECT_ROOT / "data" / "meg_segments"


@dataclass
class Config:
    source_cache_root: str = str(SOURCE_CACHE_ROOT)
    annotation_path: str = str(ANNOTATION_PATH)
    n_rois: int = 68
    n_bands: int = 4
    target_length: int = 200
    context_radius: int = 1
    dim: int = 128
    heads: int = 8
    layers: int = 4
    ffn_dim: int = 256
    contrastive_dim: int = 128
    hidden_dim: int = 128
    context_blocks: int = 3
    temporal_kernel_sizes: tuple[int, ...] = (9, 25, 25)
    temporal_dilations: tuple[int, ...] = (1, 1, 2)
    dropout: float = 0.2
    n_classes: int = 3
    class_names: tuple[str, ...] = ("negative", "neutral", "positive")
    contrastive_temperature: float = 0.1
    contrastive_loss_weight: float = 0.1
    lds_bins: int = 50
    lds_sigma: float = 0.05
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    cv_folds: int = 10
    val_subjects_per_fold: int = 1
    seed: int = 20260906
    num_workers: int = 0
    normalization_chunk_size: int = 64
    device: str = "cuda"

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        fixed = Config()
        architecture = (
            "n_rois", "n_bands", "target_length", "context_radius", "dim",
            "heads", "layers", "ffn_dim", "contrastive_dim", "hidden_dim",
            "context_blocks", "temporal_kernel_sizes", "temporal_dilations",
            "n_classes", "class_names", "dropout", "contrastive_temperature",
            "contrastive_loss_weight", "lds_bins", "lds_sigma",
        )
        for name in architecture:
            actual, expected = getattr(self, name), getattr(fixed, name)
            if isinstance(expected, tuple):
                actual = tuple(actual)
            if actual != expected:
                raise ValueError(f"SANED requires {name}={expected!r}, got {actual!r}")
        if self.epochs < 1 or self.batch_size < 4 or self.num_workers < 0:
            raise ValueError("Require epochs >= 1, batch_size >= 4, num_workers >= 0")
        if not 0 < self.min_learning_rate <= self.learning_rate:
            raise ValueError("Require 0 < min_learning_rate <= learning_rate")
        if self.normalization_chunk_size < 1 or self.weight_decay < 0 or self.seed < 0:
            raise ValueError("Invalid normalization chunk size, weight decay, or seed")
