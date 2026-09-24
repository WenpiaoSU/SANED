"""Source ROI sentence data with training-subject normalization."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .config import Config

CLASS_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}

def get_subjects(cache_root: str | Path) -> list[str]:
    return sorted(p.name.removesuffix("_source_roi.npy") for p in Path(cache_root).glob("sub-*_source_roi.npy"))


def load_annotation(path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    required = {"segment_id", "label_order", "emotion_coarse", "valence", "arousal"}
    if required - set(table.columns):
        raise ValueError(f"Annotation is missing columns: {sorted(required - set(table.columns))}")
    if "segment_type" in table:
        table = table[table.segment_type.astype(str).eq("speech")].copy()
    table = table.sort_values("label_order").reset_index(drop=True)
    if len(table) < 3 or table.segment_id.isna().any() or table.segment_id.astype(str).duplicated().any():
        raise ValueError("Require at least three uniquely identified speech sentences")
    if table.label_order.isna().any() or table.label_order.duplicated().any():
        raise ValueError("label_order must uniquely specify sentence order")
    targets = table[["valence", "arousal"]].to_numpy(dtype=np.float32)
    if not np.isfinite(targets).all() or np.any(np.abs(targets) > 1):
        raise ValueError("Valence/arousal must be finite values in [-1, 1]")
    return table


def load_roi_metadata(cache_root: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    table = pd.read_csv(Path(cache_root) / "roi_metadata.tsv", sep="\t")
    positions = table[["x", "y", "z"]].to_numpy(dtype=np.float32)
    hemispheres = table.hemisphere.astype(str).str.lower().map(
        {"lh": 0, "left": 0, "rh": 1, "right": 1}
    ).to_numpy(dtype=np.int64)
    return positions, hemispheres, table.roi_name.astype(str).tolist()


def make_fold_datasets(split: dict, cfg: Config, annotation: pd.DataFrame):
    root = Path(cfg.source_cache_root)
    groups = [split[key] for key in ("fit_subjects", "validation_subjects", "test_subjects")]
    if any(not group for group in groups):
        raise ValueError("Training, validation, and test subjects must all be nonempty")
    flat = [subject for group in groups for subject in group]
    if len(flat) != len(set(flat)) or set(flat) != set(get_subjects(root)):
        raise ValueError("Fold must partition all available subjects without overlap")
    mean, std = compute_source_normalization(groups[0], root, cfg.normalization_chunk_size)
    normalization = {subject: (mean, std) for subject in flat}
    datasets = [SourceROIContextDataset(
        make_refs(group, len(annotation), cfg.context_radius), annotation, root,
        normalization, cfg.context_radius,
    ) for group in groups]
    return (*datasets, mean, std)

@dataclass(frozen=True)
class SampleRef:
    """标识一个被试及其三句上下文的中心句子。"""

    subject: str
    center: int


class SourceROIContextDataset(Dataset):
    """从预先计算的 source-space ROI cache 加载上下文样本。

    每个被试文件必须是 ``[events, rois, bands, time]`` 的 float32 数组。
    ROI 和频段维度在所有被试间固定，避免在训练阶段引入被试或 ROI 大小偏差。
    """

    def __init__(
        self,
        refs: Iterable[SampleRef],
        annotation: pd.DataFrame,
        cache_root: Path,
        normalization: dict[str, tuple[np.ndarray, np.ndarray]],
        context_radius: int = 1,
    ) -> None:
        self.refs = list(refs)
        self.annotation = annotation
        self.cache_root = cache_root
        self.normalization = normalization
        self.context_radius = context_radius
        self._arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.refs)

    def _array(self, subject: str) -> np.ndarray:
        if subject not in self._arrays:
            self._arrays[subject] = np.load(
                self.cache_root / f"{subject}_source_roi.npy", mmap_mode="r"
            )
        return self._arrays[subject]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        ref = self.refs[index]
        arr = self._array(ref.subject)
        i0 = ref.center - self.context_radius
        i1 = ref.center + self.context_radius + 1
        x = np.asarray(arr[i0:i1], dtype=np.float32).copy()
        mean, std = self.normalization[ref.subject]
        x = (x - mean[None, :, :, None]) / std[None, :, :, None]
        row = self.annotation.iloc[ref.center]
        label = CLASS_TO_ID.get(str(row.emotion_coarse), -100)
        return {
            "x": torch.from_numpy(x),
            "class": torch.tensor(label, dtype=torch.long),
            "valence": torch.tensor(float(row.valence), dtype=torch.float32),
            "arousal": torch.tensor(float(row.arousal), dtype=torch.float32),
            "subject": ref.subject,
            "center": ref.center,
            "event_id": str(row.segment_id),
            "contrast_id": str(row.segment_id),
            "segment_id": str(row.segment_id),
        }


def compute_source_normalization(
    subjects: Iterable[str], cache_root: Path, chunk_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """汇总全部训练被试，计算一个训练折级别的 ROI x band mean/std。"""
    subjects = list(subjects)
    if not subjects:
        raise ValueError("Cannot compute source normalization without subjects")
    sums = sums2 = None
    count = 0
    for subject in subjects:
        arr = np.load(cache_root / f"{subject}_source_roi.npy", mmap_mode="r")
        if arr.ndim != 4:
            raise ValueError(f"{subject}: expected [events, rois, bands, time], got {arr.shape}")
        if sums is None:
            sums = np.zeros(arr.shape[1:3], dtype=np.float64)
            sums2 = np.zeros(arr.shape[1:3], dtype=np.float64)
        if arr.shape[1:3] != sums.shape:
            raise ValueError(f"{subject}: inconsistent source ROI/band shape {arr.shape[1:3]}")
        for start in range(0, len(arr), chunk_size):
            block = np.asarray(arr[start : start + chunk_size], dtype=np.float64)
            sums += block.sum(axis=(0, 3))
            sums2 += np.square(block).sum(axis=(0, 3))
            count += block.shape[0] * block.shape[3]
    mean64 = sums / count
    std = np.sqrt(np.maximum(sums2 / count - np.square(mean64), 1e-24))
    return mean64.astype(np.float32), np.maximum(std, 1e-12).astype(np.float32)


def validate_source_cache(cfg: Config, annotation: pd.DataFrame) -> dict:
    """验证 source-space cache 的形状和事件数，不执行隐式 source reconstruction。"""
    root = Path(cfg.source_cache_root)
    subjects = get_subjects(root)
    if not subjects:
        raise FileNotFoundError(
            f"No source ROI cache found under {root}. Expected sub-*_source_roi.npy "
            "with shape [events, rois, bands, time]. Generate the source cache before "
            "training SANED."
        )
    expected = None
    for subject in subjects:
        path = root / f"{subject}_source_roi.npy"
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 4:
            raise ValueError(f"{path}: expected [events, rois, bands, time], got {arr.shape}")
        if arr.dtype != np.float32:
            raise ValueError(f"{path}: expected float32, got {arr.dtype}")
        for start in range(0, len(arr), cfg.normalization_chunk_size):
            if not np.isfinite(arr[start:start + cfg.normalization_chunk_size]).all():
                raise ValueError(f"{path}: contains non-finite source data")
        if arr.shape[0] != len(annotation):
            raise ValueError(f"{path}: {arr.shape[0]} events but {len(annotation)} annotations")
        if expected is None:
            expected = arr.shape[1:]
        if tuple(arr.shape[1:]) != tuple(expected):
            raise ValueError(f"{path}: source shape {arr.shape[1:]} differs from {expected}")
    if expected[0] != cfg.n_rois or expected[1] != cfg.n_bands or expected[2] != cfg.target_length:
        raise ValueError(
            f"Source cache shape {expected} does not match config "
            f"(rois={cfg.n_rois}, bands={cfg.n_bands}, time={cfg.target_length})"
        )
    source_manifest_path = root / "sentence_manifest.tsv"
    if not source_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {source_manifest_path}; source cache must preserve the exact speech sentence order"
        )
    source_manifest = pd.read_csv(source_manifest_path, sep="\t")
    if "segment_id" not in source_manifest:
        raise ValueError("sentence_manifest.tsv must contain segment_id")
    expected_segment_ids = annotation.segment_id.astype(str).tolist()
    source_segment_ids = source_manifest.segment_id.astype(str).tolist()
    if source_segment_ids != expected_segment_ids:
        raise ValueError(
            "Source sentence_manifest.tsv segment_id order/content does not match speech annotation"
        )
    roi_metadata_path = root / "roi_metadata.tsv"
    if not roi_metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {roi_metadata_path}; source mode requires columns "
            "roi_index, roi_name, hemisphere, x, y, z"
        )
    roi_metadata = pd.read_csv(roi_metadata_path, sep="\t")
    required = {"roi_index", "roi_name", "hemisphere", "x", "y", "z"}
    missing_columns = sorted(required - set(roi_metadata.columns))
    if missing_columns:
        raise ValueError(f"{roi_metadata_path}: missing columns {missing_columns}")
    if len(roi_metadata) != cfg.n_rois:
        raise ValueError(
            f"{roi_metadata_path}: expected {cfg.n_rois} rows, got {len(roi_metadata)}"
        )
    if roi_metadata.roi_index.astype(int).tolist() != list(range(cfg.n_rois)):
        raise ValueError("roi_metadata.tsv roi_index must be contiguous and zero-based")
    if roi_metadata.roi_name.isna().any() or roi_metadata.roi_name.duplicated().any():
        raise ValueError("ROI names must be nonempty and unique")
    if not set(roi_metadata.hemisphere.astype(str).str.lower()) <= {"lh", "rh", "left", "right"}:
        raise ValueError("roi_metadata.tsv hemisphere values must be lh/rh or left/right")
    if not np.isfinite(roi_metadata[["x", "y", "z"]].to_numpy(dtype=float)).all():
        raise ValueError("roi_metadata.tsv contains non-finite coordinates")
    metadata = {
        "subjects": subjects,
        "shape_per_subject": [len(annotation), *expected],
        "root": str(root),
        "input_mode": "source",
        "roi_metadata": str(roi_metadata_path),
        "sentence_manifest": str(source_manifest_path),
    }
    return metadata


def make_refs(subjects: Iterable[str], n_segments: int, context_radius: int) -> list[SampleRef]:
    """创建中心句子引用，并确保上下文索引不越界。"""
    refs = []
    for subject in subjects:
        refs.extend(SampleRef(subject, i) for i in range(context_radius, n_segments - context_radius))
    return refs
