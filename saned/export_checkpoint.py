"""Package trained SANED weights with their exact fold and preprocessing metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import load_weights, make_checkpoint, save_checkpoint
from .config import Config
from .data import compute_source_normalization, get_subjects, load_annotation, load_roi_metadata, validate_source_cache
from .models import SANED


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Selected best.pt containing a raw state_dict")
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--split", type=Path, required=True, help="JSON with fit_subjects, validation_subjects, test_subjects")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    cfg = Config(source_cache_root=str(args.source_cache_root.resolve()), device="cpu")
    cfg.annotation_path = str((args.annotation or args.source_cache_root / "sentence_manifest.tsv").resolve())
    annotation = load_annotation(cfg.annotation_path)
    validate_source_cache(cfg, annotation)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    keys = ("fit_subjects", "validation_subjects", "test_subjects")
    if any(not isinstance(split.get(key), list) or not split[key] for key in keys):
        raise ValueError(f"Split JSON must contain nonempty lists: {keys}")
    subjects = [subject for key in keys for subject in split[key]]
    if len(subjects) != len(set(subjects)) or set(subjects) != set(get_subjects(args.source_cache_root)):
        raise ValueError("Split must partition the exact source-cache subject set")
    metadata = load_roi_metadata(args.source_cache_root)
    model = SANED(cfg, metadata[0], metadata[1])
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    load_weights(model, state)
    mean, std = compute_source_normalization(
        split["fit_subjects"], args.source_cache_root, cfg.normalization_chunk_size
    )
    checkpoint = make_checkpoint(model, cfg, mean, std, metadata, split, None, None)
    checkpoint["training_settings_verified"] = False
    checkpoint["provenance"] = "Selected trained weights; normalization recomputed from the supplied fitting subjects"
    save_checkpoint(checkpoint, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
