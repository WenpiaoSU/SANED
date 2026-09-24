"""Predict center-sentence emotion using a complete SANED checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_json
from .config import PROJECT_ROOT
from .data import SourceROIContextDataset, get_subjects, load_annotation, load_roi_metadata, make_refs, validate_source_cache
from .metrics import aggregate_sentence_predictions, evaluate, prediction_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, default=PROJECT_ROOT / "data" / "source_cache")
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--subjects", help="Comma-separated IDs; defaults to checkpoint test subjects")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "predictions.tsv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    model, cfg, checkpoint, mean, std = load_checkpoint(args.checkpoint, args.device)
    cfg.source_cache_root = str(args.source_cache_root.resolve())
    root = Path(cfg.source_cache_root)
    manifest = pd.read_csv(root / "sentence_manifest.tsv", sep="\t")
    annotation_path = args.annotation or root / "sentence_manifest.tsv"
    annotation_table = pd.read_csv(annotation_path, sep="\t")
    has_labels = {"emotion_coarse", "valence", "arousal"} <= set(annotation_table.columns)
    if has_labels:
        annotation = load_annotation(annotation_path)
    else:
        if args.annotation is not None:
            raise ValueError("Explicit annotation must contain emotion_coarse, valence, and arousal")
        if ("segment_id" not in manifest or len(manifest) < 3
                or manifest.segment_id.isna().any() or manifest.segment_id.astype(str).duplicated().any()):
            raise ValueError("Require at least three uniquely identified sentences in the manifest")
        # Labels are placeholders for the shared data loader and are omitted from output.
        annotation = manifest.assign(emotion_coarse="unlabeled", valence=0.0, arousal=0.0)
    validate_source_cache(cfg, annotation)
    positions, hemispheres, names = load_roi_metadata(root)
    if (names != checkpoint["roi_names"]
            or not np.array_equal(hemispheres, checkpoint["roi_hemispheres"].numpy())
            or not np.allclose(positions, checkpoint["roi_positions"].numpy(), rtol=1e-6, atol=1e-8)):
        raise ValueError("Cache ROI order/geometry differs from the checkpoint")
    subjects = ([s.strip() for s in args.subjects.split(",") if s.strip()]
                if args.subjects else checkpoint["split"]["test_subjects"])
    if not subjects or len(subjects) != len(set(subjects)) or not set(subjects) <= set(get_subjects(root)):
        raise ValueError("Requested subjects must be unique and present in the source cache")
    ds = SourceROIContextDataset(make_refs(subjects, len(annotation), cfg.context_radius),
                                 annotation, root, {s: (mean, std) for s in subjects}, cfg.context_radius)
    predictions = aggregate_sentence_predictions(evaluate(
        model, DataLoader(ds, batch_size=args.batch_size, shuffle=False), torch.device(args.device)
    ))
    predictions["emotion"] = predictions.class_pred.map(dict(enumerate(cfg.class_names)))
    if has_labels:
        c, v, a = prediction_metrics(predictions)
        save_json({"classification": c, "valence": v, "arousal": a}, args.output.with_suffix(".metrics.json"))
    else:
        predictions = predictions.drop(columns=["class_true", "valence_true", "arousal_true"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, sep="\t", index=False)
    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
