"""Train and evaluate SANED with subject-independent cross-validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .checkpoint import make_checkpoint, save_checkpoint, save_json
from .config import Config, PROJECT_ROOT
from .data import get_subjects, load_annotation, load_roi_metadata, make_fold_datasets, validate_source_cache
from .losses import LDSWeightLookup, masked_cross_entropy, multi_positive_infonce, weighted_mse
from .metrics import aggregate_sentence_predictions, evaluate, prediction_metrics
from .models import SANED
from .sampling import EventBalancedBatchSampler
from .training_utils import build_cosine_scheduler, make_cv_splits, seed_everything


def run_fold(split: dict, cfg: Config, annotation: pd.DataFrame, metadata, output: Path) -> dict:
    fold = split["fold"]
    folder = output / "folds" / f"fold_{fold:02d}"
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {folder}; use a new --output-root")
    folder.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed + fold)
    device = torch.device(cfg.device)
    train_ds, val_ds, test_ds, mean, std = make_fold_datasets(split, cfg, annotation)
    sampler = EventBalancedBatchSampler(
        train_ds.refs, cfg.batch_size, cfg.seed + fold,
        balance_ids={i: str(row.segment_id) for i, row in annotation.iterrows()},
    )
    loader_options = {"num_workers": cfg.num_workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_ds, batch_sampler=sampler, **loader_options)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **loader_options)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **loader_options)
    model = SANED(cfg, metadata[0], metadata[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = build_cosine_scheduler(optimizer, cfg, len(train_loader))
    centers = annotation.loc[annotation.index.isin({ref.center for ref in train_ds.refs})]
    train_values = centers.drop_duplicates("segment_id")[["valence", "arousal"]].to_numpy(dtype=np.float32)
    lds_v = LDSWeightLookup(train_values[:, 0], cfg.lds_bins, cfg.lds_sigma)
    lds_a = LDSWeightLookup(train_values[:, 1], cfg.lds_bins, cfg.lds_sigma)
    save_json(split, folder / "split.json")
    best_score, best_epoch = float("inf"), 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = []
        learning_rate = optimizer.param_groups[0]["lr"]
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            targets = {key: batch[key].to(device) for key in ("class", "valence", "arousal")}
            pred = model(x)
            classification = masked_cross_entropy(pred["class_logits"], targets["class"])
            valence = weighted_mse(pred["valence"], targets["valence"], lds_v.tensor(targets["valence"]))
            arousal = weighted_mse(pred["arousal"], targets["arousal"], lds_a.tensor(targets["arousal"]))
            contrastive = multi_positive_infonce(
                pred["embedding"], list(batch["subject"]), list(batch["segment_id"]),
                temperature=cfg.contrastive_temperature,
            )
            loss = classification + valence + arousal + cfg.contrastive_loss_weight * contrastive
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in fold {fold}, epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            running.append([float(value.detach()) for value in (loss, classification, valence, arousal, contrastive)])
        predictions = aggregate_sentence_predictions(evaluate(model, val_loader, device))
        c, v, a = prediction_metrics(predictions)
        score = v["mse"] + a["mse"]
        history.append({
            "epoch": epoch, "learning_rate": learning_rate,
            "loss": dict(zip(("total", "classification", "valence", "arousal", "contrastive"),
                             np.mean(running, axis=0).tolist())),
            "classification": c, "valence": v, "arousal": a,
        })
        if score < best_score:
            best_score, best_epoch = score, epoch
            checkpoint = make_checkpoint(model, cfg, mean, std, metadata, split, epoch, score)
            save_checkpoint(checkpoint, folder / "best.pt")
        save_json(history, folder / "history.json")
        print(f"fold={fold:02d} epoch={epoch:03d}/{cfg.epochs} "
              f"loss={np.mean(running, axis=0)[0]:.5f} val_score={score:.5f}", flush=True)
    if best_epoch == 0:
        raise RuntimeError("No finite validation score was obtained")
    best = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"], strict=True)
    predictions = aggregate_sentence_predictions(evaluate(model, test_loader, device))
    predictions.to_csv(folder / "test_predictions.tsv", sep="\t", index=False)
    c, v, a = prediction_metrics(predictions)
    result = {
        "fold": fold, "model_name": "SANED", "test_subjects": split["test_subjects"],
        "n_test": len(predictions), "classification": c, "valence": v, "arousal": a,
        "best_epoch": best_epoch, "best_validation_score": best_score,
        "selection_metric": "valence_mse+arousal_mse",
    }
    save_json(result, folder / "metrics.json")
    return result


def summarize(output: Path, splits: list[dict]) -> None:
    rows, predictions = [], []
    for split in splits:
        folder = output / "folds" / f"fold_{split['fold']:02d}"
        result = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        if result["test_subjects"] != split["test_subjects"]:
            raise ValueError(f"Inconsistent test subjects in {folder}")
        row = {"fold": split["fold"], "n_test": result["n_test"]}
        for task in ("classification", "valence", "arousal"):
            row.update({f"{task}_{key}": val for key, val in result[task].items()
                        if isinstance(val, (int, float)) or val is None})
        rows.append(row)
        predictions.append(pd.read_csv(folder / "test_predictions.tsv", sep="\t"))
    table = pd.DataFrame(rows)
    table.to_csv(output / "fold_metrics.tsv", sep="\t", index=False)
    pooled = pd.concat(predictions, ignore_index=True)
    pooled.to_csv(output / "test_predictions.tsv", sep="\t", index=False)
    c, v, a = prediction_metrics(pooled)
    metrics = table.drop(columns=["fold", "n_test"])
    save_json({
        "n_folds": len(splits), "fold_metrics_mean": metrics.mean(numeric_only=True).to_dict(),
        "fold_metrics_std": metrics.std(ddof=1, numeric_only=True).to_dict(),
        "pooled": {"classification": c, "valence": v, "arousal": a},
    }, output / "summary.json")


def main() -> None:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache-root", type=Path, default=Path(defaults.source_cache_root))
    parser.add_argument("--annotation", type=Path, help="Defaults to source-cache-root/sentence_manifest.tsv")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--folds", help="Comma-separated fold numbers, e.g. 1,2,3; default: all ten")
    parser.add_argument("--summarize-only", action="store_true", help="Aggregate all completed folds")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if args.summarize_only:
        splits = json.loads((output / "splits.json").read_text(encoding="utf-8"))
        summarize(output, splits)
        return
    root = args.source_cache_root.resolve()
    cfg = Config(
        source_cache_root=str(root), annotation_path=str((args.annotation or root / "sentence_manifest.tsv").resolve()),
        device=args.device, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers,
    )
    cfg.validate()
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install a suitable PyTorch build or use --device cpu")
    annotation = load_annotation(cfg.annotation_path)
    validate_source_cache(cfg, annotation)
    metadata = load_roi_metadata(root)
    splits = make_cv_splits(get_subjects(root), cfg.cv_folds, cfg.val_subjects_per_fold, cfg.seed)
    selected = set(range(1, cfg.cv_folds + 1)) if args.folds is None else {int(x) for x in args.folds.split(",")}
    if not selected or not selected <= {split["fold"] for split in splits}:
        raise ValueError("--folds must contain fold numbers 1 through 10")
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("config.json", cfg.to_dict()), ("splits.json", splits)):
        path = output / name
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved != json.loads(json.dumps(value)):
                raise ValueError(f"Existing {path} belongs to another configuration; use a new output directory")
        else:
            save_json(value, path)
    for split in splits:
        if split["fold"] in selected:
            run_fold(split, cfg, annotation, metadata, output)
    if all((output / "folds" / f"fold_{s['fold']:02d}" / "metrics.json").exists() for s in splits):
        summarize(output, splits)


if __name__ == "__main__":
    main()
