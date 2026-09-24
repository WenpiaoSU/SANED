"""Generate sentence-level SANED source ROI caches.

The script reconstructs continuous source activity with an ``fsaverage`` ico-5
source space and a common-reference spherical forward model using each
subject's sensor geometry.  It then extracts
Desikan--Killiany (``aparc``) ROI time courses, computes four Hilbert envelopes,
and resamples the 738 speech sentences to the fixed SANED length (200 samples).

This is an explicit, auditable approximation for the current dataset: no
individual MRI/BEM/coregistration is available, so the output must not be
interpreted as an individual-anatomy source localisation result.  It is still a
real forward/inverse reconstruction and is never a reshape of the 64 sensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd
from scipy.signal import hilbert

from .config import (
    MEG_SEGMENT_ROOT,
    SOURCE_CACHE_ROOT,
)
from .data import validate_source_cache
from .preprocessing import resample_segment


LOGGER = logging.getLogger("saned_source_cache")
TARGET_LENGTH = 200
N_ROIS = 68
BANDS: tuple[tuple[str, float, float], ...] = (
    ("theta", 4.0, 7.0),
    ("alpha", 8.0, 12.0),
    ("beta", 13.0, 29.0),
    ("low_gamma", 30.0, 38.0),
)
# The official FIF files share the same 64-channel device geometry, but most
# files do not contain the head-shape digitisation kinds required by MNE's
# ``r0='auto'`` fit.  This reference origin was fitted once from sub-31's
# audited head points and is reused consistently for all subjects.
REFERENCE_SPHERE_ORIGIN_M = np.array([0.00282627, 0.00560018, -0.03625677], dtype=float)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def configure_logging(log_path: Path, verbose: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.FileHandler(log_path, encoding="utf-8")]
    stream_level = logging.INFO if verbose else logging.WARNING
    handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    handlers[-1].setLevel(stream_level)


def parse_subjects(value: str | None, derivatives_root: Path, segment_root: Path) -> list[str]:
    if value:
        subjects = [item.strip() for item in value.split(",") if item.strip()]
    else:
        subjects = sorted(
            p.name.split("_")[0]
            for p in segment_root.glob("sub-*_task-baba_sentence_meg.npy")
        )
    if not subjects:
        raise FileNotFoundError(f"No sentence MEG arrays found under {segment_root}")
    for subject in subjects:
        fif = derivatives_root / subject / "meg" / f"{subject}_task-baba_desc-preproc_meg.fif"
        npy = segment_root / f"{subject}_task-baba_sentence_meg.npy"
        tsv = segment_root / f"{subject}_task-baba_sentence_meg.tsv"
        missing = [str(path) for path in (fif, npy, tsv) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{subject}: missing required files: {missing}")
    return subjects


def load_speech_annotation(annotation_path: Path) -> pd.DataFrame:
    annotation = pd.read_csv(annotation_path, sep="\t")
    if "segment_type" in annotation:
        annotation = annotation[annotation.segment_type.astype(str).eq("speech")].copy()
    if "label_order" not in annotation or "segment_id" not in annotation:
        raise ValueError("Annotation must contain label_order and segment_id")
    annotation = annotation.sort_values("label_order").reset_index(drop=True)
    if len(annotation) != 738:
        raise ValueError(f"Expected 738 speech annotations, got {len(annotation)}")
    if annotation.segment_id.astype(str).duplicated().any():
        raise ValueError("Speech annotation contains duplicated segment_id values")
    return annotation


def validate_segment_table(table: pd.DataFrame, sensor_array: np.ndarray, annotation: pd.DataFrame, subject: str) -> None:
    required = {"meg_start_sample", "meg_end_sample", "n_samples", "meg_onset", "meg_offset"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{subject}: sentence table missing columns {missing}")
    if len(table) != len(annotation):
        raise ValueError(f"{subject}: {len(table)} sentence rows but {len(annotation)} annotations")
    starts = table.meg_start_sample.to_numpy(dtype=np.int64)
    ends = table.meg_end_sample.to_numpy(dtype=np.int64)
    lengths = table.n_samples.to_numpy(dtype=np.int64)
    if starts[0] != 0 or ends[-1] != sensor_array.shape[1]:
        raise ValueError(f"{subject}: sentence bounds do not cover the sensor array exactly")
    if np.any(ends <= starts) or np.any(lengths != ends - starts):
        raise ValueError(f"{subject}: invalid sentence sample bounds")
    if np.any(starts[1:] != ends[:-1]):
        raise ValueError(f"{subject}: sentence bounds are not contiguous")
    if sensor_array.ndim != 2 or sensor_array.shape[0] != 64:
        raise ValueError(f"{subject}: expected sensor array [64, time], got {sensor_array.shape}")


def continuous_sentence_bounds(table: pd.DataFrame, sfreq: float, n_times: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.rint(table.meg_onset.to_numpy(dtype=np.float64) * sfreq).astype(np.int64)
    lengths = table.n_samples.to_numpy(dtype=np.int64)
    stops = starts + lengths
    if np.any(starts < 0) or np.any(stops > n_times) or np.any(stops <= starts):
        raise ValueError("Sentence timing falls outside the official continuous FIF")
    return starts, stops


def audit_sensor_alignment(
    continuous: np.ndarray,
    concatenated: np.ndarray,
    table: pd.DataFrame,
    sfreq: float,
    subject: str,
) -> float:
    """Prove that sentence arrays are exact slices of the official continuous FIF."""
    starts, stops = continuous_sentence_bounds(table, sfreq, continuous.shape[1])
    max_abs_difference = 0.0
    for row_index, (start, stop, row) in enumerate(
        zip(starts, stops, table.itertuples(index=False), strict=True)
    ):
        expected = np.asarray(
            concatenated[:, int(row.meg_start_sample) : int(row.meg_end_sample)]
        )
        actual = continuous[:, start:stop]
        if actual.shape != expected.shape:
            raise ValueError(f"{subject}: sentence {row_index} has inconsistent shapes")
        difference = float(np.max(np.abs(actual - expected)))
        max_abs_difference = max(max_abs_difference, difference)
    if max_abs_difference > 1e-18:
        raise ValueError(
            f"{subject}: official FIF and sentence cache differ (max abs={max_abs_difference:.3e})"
        )
    return max_abs_difference


def load_fsaverage(fsaverage_root: Path):
    src_path = fsaverage_root / "bem" / "fsaverage-ico-5-src.fif"
    trans_path = fsaverage_root / "bem" / "fsaverage-trans.fif"
    if not src_path.exists() or not trans_path.exists():
        raise FileNotFoundError(f"Missing fsaverage source/transform under {fsaverage_root}")
    subjects_dir = fsaverage_root.parent
    labels = mne.read_labels_from_annot(
        "fsaverage", parc="aparc", subjects_dir=str(subjects_dir), verbose="ERROR"
    )
    labels = [label for label in labels if not label.name.startswith("unknown-")]
    if len(labels) != N_ROIS:
        raise ValueError(f"Expected {N_ROIS} aparc labels after removing unknown, got {len(labels)}")
    src = mne.read_source_spaces(str(src_path), verbose="ERROR")
    roi_rows = []
    for roi_index, label in enumerate(labels):
        xyz = np.asarray(label.pos, dtype=np.float64).mean(axis=0)
        roi_rows.append(
            {
                "roi_index": roi_index,
                "roi_name": label.name,
                "hemisphere": label.hemi,
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
                "coordinate_frame": "mri_surface_ras_m",
                "n_template_vertices": int(len(label.vertices)),
            }
        )
    return src, labels, pd.DataFrame(roi_rows), src_path, trans_path


def make_inverse(info: mne.Info, src, trans_path: Path, n_jobs: int):
    # ``head_radius=None`` selects MNE's single-layer MEG sphere.  The
    # multi-shell fit to the OPM digitisation would use an ~84 mm inner shell,
    # which clips a substantial fraction of the fsaverage cortex and leaves
    # some aparc labels empty.  A single-layer MEG sphere keeps all ico-5
    # cortical vertices while retaining a valid sensor-to-source forward model.
    sphere = mne.make_sphere_model(
        info=info, r0=REFERENCE_SPHERE_ORIGIN_M, head_radius=None, verbose="ERROR"
    )
    forward = mne.make_forward_solution(
        info,
        trans=str(trans_path),
        src=src,
        bem=sphere,
        meg=True,
        eeg=False,
        mindist=5.0,
        n_jobs=n_jobs,
        verbose="ERROR",
    )
    # No empty-room recording is available.  The documented MNE ad-hoc
    # magnetometer covariance keeps the inverse protocol deterministic and
    # makes the limitation explicit in the run metadata.
    noise_cov = mne.make_ad_hoc_cov(info, verbose="ERROR")
    inverse = mne.minimum_norm.make_inverse_operator(
        info,
        forward,
        noise_cov,
        loose=0.2,
        depth=0.8,
        verbose="ERROR",
    )
    return forward, inverse, sphere


def reconstruct_roi_timeseries(
    sensor_array: np.ndarray,
    info: mne.Info,
    inverse,
    labels,
    output_path: Path,
    chunk_samples: int,
    subject: str,
) -> tuple[int, int]:
    n_samples = int(sensor_array.shape[1])
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    roi_memmap = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(N_ROIS, n_samples)
    )
    try:
        for start in range(0, n_samples, chunk_samples):
            stop = min(start + chunk_samples, n_samples)
            LOGGER.info("%s source chunk %d:%d / %d", subject, start, stop, n_samples)
            chunk = np.asarray(sensor_array[:, start:stop], dtype=np.float64)
            raw = mne.io.RawArray(chunk, info.copy(), first_samp=start, verbose="ERROR")
            stc = mne.minimum_norm.apply_inverse_raw(
                raw,
                inverse,
                lambda2=1.0 / 9.0,
                method="dSPM",
                pick_ori="normal",
                verbose="ERROR",
            )
            roi = mne.extract_label_time_course(
                stc,
                labels,
                inverse["src"],
                mode="mean_flip",
                allow_empty=False,
                verbose="ERROR",
            )
            if roi.shape != (N_ROIS, stop - start):
                raise RuntimeError(f"{subject}: unexpected ROI block shape {roi.shape}")
            roi_memmap[:, start:stop] = roi.astype(np.float32, copy=False)
            roi_memmap.flush()
        del roi_memmap
        os.replace(temporary, output_path)
    except Exception:
        del roi_memmap
        temporary.unlink(missing_ok=True)
        raise
    return N_ROIS, n_samples


def build_source_cache(
    roi_timeseries_path: Path,
    segment_table: pd.DataFrame,
    output_path: Path,
    sfreq: float,
    target_length: int,
    subject: str,
) -> dict:
    roi_ts = np.load(roi_timeseries_path, mmap_mode="r")
    n_events = len(segment_table)
    starts, stops = continuous_sentence_bounds(segment_table, sfreq, roi_ts.shape[1])
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    cache = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(n_events, N_ROIS, len(BANDS), target_length)
    )
    band_stats = []
    try:
        for band_index, (name, low, high) in enumerate(BANDS):
            LOGGER.info("%s filtering %s %.1f-%.1f Hz", subject, name, low, high)
            filtered = mne.filter.filter_data(
                np.asarray(roi_ts, dtype=np.float64),
                sfreq=sfreq,
                l_freq=low,
                h_freq=high,
                method="fir",
                phase="zero-double",
                n_jobs=1,
                verbose="ERROR",
            )
            envelope = np.abs(hilbert(filtered, axis=-1)).astype(np.float32)
            for event_index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
                cache[event_index, :, band_index, :] = resample_segment(
                    envelope[:, start:stop], target_length
                )
            cache.flush()
            band_stats.append(
                {
                    "band": name,
                    "low_hz": low,
                    "high_hz": high,
                    "mean": float(np.mean(envelope)),
                    "std": float(np.std(envelope)),
                    "min": float(np.min(envelope)),
                    "max": float(np.max(envelope)),
                }
            )
            del filtered, envelope
        del cache
        os.replace(temporary, output_path)
    except Exception:
        del cache
        temporary.unlink(missing_ok=True)
        raise
    arr = np.load(output_path, mmap_mode="r")
    finite = bool(np.isfinite(arr).all())
    nonzero = int(np.count_nonzero(arr))
    roi_std = np.asarray(arr, dtype=np.float64).std(axis=(0, 3))
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite": finite,
        "nonzero_values": nonzero,
        "global_mean": float(np.mean(arr)),
        "global_std": float(np.std(arr)),
        "roi_band_std_min": float(np.min(roi_std)),
        "roi_band_std_max": float(np.max(roi_std)),
        "bands": band_stats,
    }


def archive_existing(path: Path, archive_root: Path) -> None:
    if not path.exists():
        return
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = archive_root / f"{path.name}.{stamp}"
    suffix = 1
    while target.exists():
        target = archive_root / f"{path.name}.{stamp}.{suffix}"
        suffix += 1
    shutil.move(str(path), str(target))
    LOGGER.warning("Archived existing cache %s -> %s", path, target)


def write_subject_status(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    pd.DataFrame(rows).to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def process_subject(
    subject: str,
    args,
    annotation: pd.DataFrame,
    src,
    labels,
    output_root: Path,
    derivatives_root: Path,
    segment_root: Path,
) -> dict:
    started = time.time()
    final_path = output_root / f"{subject}_source_roi.npy"
    subject_dir = output_root / "intermediate" / subject
    segment_path = segment_root / f"{subject}_task-baba_sentence_meg.npy"
    table_path = segment_root / f"{subject}_task-baba_sentence_meg.tsv"
    fif_path = derivatives_root / subject / "meg" / f"{subject}_task-baba_desc-preproc_meg.fif"
    try:
        if final_path.exists() and not args.overwrite:
            arr = np.load(final_path, mmap_mode="r")
            if (
                tuple(arr.shape) == (len(annotation), N_ROIS, len(BANDS), args.target_length)
                and arr.dtype == np.float32
                and np.isfinite(arr).all()
                and np.count_nonzero(arr) > 0
            ):
                return {
                    "subject": subject,
                    "status": "skipped_existing",
                    "seconds": 0.0,
                    "output": str(final_path),
                    "shape": list(arr.shape),
                    "error": "",
                }
            raise ValueError(f"Existing cache has unexpected shape/dtype: {arr.shape} {arr.dtype}")
        if final_path.exists() and args.overwrite:
            archive_existing(final_path, output_root / "archive")

        if args.overwrite and subject_dir.exists():
            archive_existing(subject_dir, output_root / "archive" / "intermediate")
        subject_dir.mkdir(parents=True, exist_ok=True)

        sensor_segments = np.load(segment_path, mmap_mode="r")
        segment_table = pd.read_csv(table_path, sep="\t")
        validate_segment_table(segment_table, sensor_segments, annotation, subject)
        epochs = mne.read_epochs(str(fif_path), preload=True, verbose="ERROR")
        info = epochs.info.copy()
        if abs(float(info["sfreq"]) - float(args.sfreq)) > 1e-6:
            raise ValueError(f"{subject}: expected {args.sfreq} Hz FIF info, got {info['sfreq']}")
        if len(info["ch_names"]) != sensor_segments.shape[0]:
            raise ValueError(f"{subject}: FIF channels and sentence array disagree")
        continuous = epochs.get_data(copy=False)[0]
        max_abs_difference = audit_sensor_alignment(
            continuous,
            sensor_segments,
            segment_table,
            float(info["sfreq"]),
            subject,
        )
        forward, inverse, sphere = make_inverse(info, src, args.trans_path, args.n_jobs)
        forward_path = subject_dir / f"{subject}-fsaverage-sphere-fwd.fif"
        inverse_path = subject_dir / f"{subject}-fsaverage-sphere-inv.fif"
        forward_temporary = subject_dir / f".{subject}.partial-fwd.fif"
        inverse_temporary = subject_dir / f".{subject}.partial-inv.fif"
        mne.write_forward_solution(str(forward_temporary), forward, overwrite=True)
        os.replace(forward_temporary, forward_path)
        mne.minimum_norm.write_inverse_operator(str(inverse_temporary), inverse, overwrite=True)
        os.replace(inverse_temporary, inverse_path)
        sphere_path = subject_dir / "sphere_model.json"
        sphere_path.write_text(
            json.dumps(
                {
                    "r0_m": np.asarray(sphere["r0"], dtype=float).tolist(),
                    "origin_source": "reference sub-31 head-shape fit",
                    "layers": [
                        {"rad_m": float(layer["rad"]), "sigma": float(layer["sigma"])}
                        for layer in sphere["layers"]
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        roi_ts_path = subject_dir / f"{subject}_roi_dspm_continuous.npy"
        reconstruct_roi_timeseries(
            continuous,
            info,
            inverse,
            labels,
            roi_ts_path,
            args.chunk_samples,
            subject,
        )
        qc = build_source_cache(
            roi_ts_path,
            segment_table,
            final_path,
            sfreq=float(info["sfreq"]),
            target_length=args.target_length,
            subject=subject,
        )
        if qc["shape"] != [len(annotation), N_ROIS, len(BANDS), args.target_length] or not qc["finite"] or qc["nonzero_values"] == 0:
            raise ValueError(f"{subject}: cache QC failed: {qc}")
        qc["sensor_alignment_max_abs_difference"] = max_abs_difference
        qc["continuous_fif_shape"] = list(continuous.shape)
        (subject_dir / "qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "subject": subject,
            "status": "completed",
            "seconds": round(time.time() - started, 3),
            "output": str(final_path),
            "shape": qc["shape"],
            "error": "",
        }
    except Exception as exc:
        LOGGER.exception("%s failed", subject)
        return {
            "subject": subject,
            "status": "failed",
            "seconds": round(time.time() - started, 3),
            "output": str(final_path),
            "shape": "",
            "error": repr(exc),
        }


def write_manifests(output_root: Path, annotation: pd.DataFrame, args, roi_metadata: pd.DataFrame, subjects: Iterable[str]) -> None:
    manifest_columns = [
        "label_order", "segment_id", "segment_type", "sentence_id", "onset", "offset", "duration",
        "emotion_coarse", "valence", "arousal"
    ]
    annotation[manifest_columns].to_csv(output_root / "sentence_manifest.tsv", sep="\t", index=False)
    roi_metadata.to_csv(output_root / "roi_metadata.tsv", sep="\t", index=False)
    config = {
        "created_at_utc": utc_now(),
        "command": " ".join([sys.executable, *sys.argv]),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "mne": mne.__version__,
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "subjects": list(subjects),
        "annotation_path": str(args.annotation),
        "annotation_sha256": sha256_file(args.annotation),
        "segment_root": str(args.segment_root),
        "derivatives_root": str(args.derivatives_root),
        "fsaverage_root": str(args.fsaverage_root),
        "source_protocol": "fsaverage ico-5 + common-reference spherical forward + dSPM",
        "source_space": "fsaverage-ico-5-src.fif",
        "forward_transform": str(args.trans_path),
        "conductor_model": "single-layer MEG sphere (head_radius=None; all fsaverage ico-5 vertices retained)",
        "sphere_origin_m": REFERENCE_SPHERE_ORIGIN_M.tolist(),
        "sphere_origin_source": "sub-31 head-shape fit; reused because other FIF files lack >=4 fit points",
        "noise_covariance": "mne.make_ad_hoc_cov (no empty-room recording available)",
        "inverse_lambda2": 1.0 / 9.0,
        "inverse_method": "dSPM",
        "inverse_orientation": "loose=0.2, surface-normal component",
        "atlas": "fsaverage aparc, unknown labels removed, 68 ROIs, mean_flip extraction",
        "bands": [dict(name=name, low_hz=low, high_hz=high) for name, low, high in BANDS],
        "target_length": args.target_length,
        "chunk_samples": args.chunk_samples,
        "n_jobs": args.n_jobs,
        "coordinate_units": "meters in fsaverage MRI surface-RAS frame",
        "approximation_warning": "Not individual MRI/BEM/coregistration; do not interpret as individual anatomy.",
    }
    (output_root / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "run_command.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", default=None, help="Comma-separated subjects, e.g. sub-31; default: all sentence arrays")
    parser.add_argument("--output-root", type=Path, default=SOURCE_CACHE_ROOT)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--segment-root", type=Path, default=MEG_SEGMENT_ROOT)
    parser.add_argument(
        "--derivatives-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fsaverage-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--chunk-samples", type=int, default=5000)
    parser.set_defaults(target_length=TARGET_LENGTH)
    parser.add_argument("--sfreq", type=float, default=100.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.n_jobs < 1 or args.chunk_samples < 1 or args.target_length < 2:
        raise ValueError("n_jobs, chunk_samples must be positive and target_length >= 2")
    args.annotation = args.annotation.resolve()
    args.segment_root = args.segment_root.resolve()
    args.derivatives_root = args.derivatives_root.resolve()
    args.fsaverage_root = args.fsaverage_root.resolve()
    args.output_root = args.output_root.resolve()
    args.trans_path = args.fsaverage_root / "bem" / "fsaverage-trans.fif"
    args.output_root.mkdir(parents=True, exist_ok=True)
    configure_logging(args.output_root / "logs" / "run.log", args.verbose)
    LOGGER.info("Starting source cache generation in %s", args.output_root)
    annotation = load_speech_annotation(args.annotation)
    subjects = parse_subjects(args.subjects, args.derivatives_root, args.segment_root)
    src, labels, roi_metadata, _, _ = load_fsaverage(args.fsaverage_root)
    write_manifests(args.output_root, annotation, args, roi_metadata, subjects)
    status_rows: list[dict] = []
    status_path = args.output_root / "subject_status.tsv"
    for subject in subjects:
        result = process_subject(
            subject,
            args,
            annotation,
            src,
            labels,
            args.output_root,
            args.derivatives_root,
            args.segment_root,
        )
        status_rows.append(result)
        write_subject_status(status_path, status_rows)
        LOGGER.info("%s status=%s (completed=%d failed=%d skipped=%d)", subject, result["status"], sum(r["status"] == "completed" for r in status_rows), sum(r["status"] == "failed" for r in status_rows), sum(r["status"] == "skipped_existing" for r in status_rows))
    summary = {
        "created_at_utc": utc_now(),
        "output_root": str(args.output_root),
        "subjects_requested": subjects,
        "completed": [r["subject"] for r in status_rows if r["status"] == "completed"],
        "skipped_existing": [r["subject"] for r in status_rows if r["status"] == "skipped_existing"],
        "failed": [r["subject"] for r in status_rows if r["status"] == "failed"],
        "status_table": str(status_path),
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if summary["failed"]:
        LOGGER.error("Source cache generation finished with failures: %s", summary["failed"])
        return 1
    from .config import Config

    cfg = Config(source_cache_root=str(args.output_root), annotation_path=str(args.annotation))
    validation = validate_source_cache(cfg, load_speech_annotation(args.annotation))
    (args.output_root / "cache_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Source cache generation completed successfully for %d subjects", len(subjects))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
