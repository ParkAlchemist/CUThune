from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from tqdm import tqdm

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3] if len(THIS_FILE.parents) > 3 else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.src.chiptune_shared.cqt.features import (
    audio_stats,
    list_audio_files,
    load_audio,
    stable_id,
    trim_silence,
    peak_normalize,
    rms_normalize,
    write_jsonl,
)

CACHE_VERSION = 4
EPS = 1e-8


@dataclass(frozen=True)
class CQTSettings:
    sample_rate: int
    hop_length: int
    n_bins: int
    bins_per_octave: int
    fmin: float
    filter_scale: float
    norm: float
    sparsity: float
    window: str
    scale: bool
    pad_mode: str


@dataclass(frozen=True)
class PreprocessSettings:
    trim_silence: bool
    trim_top_db: float
    trim_margin_seconds: float
    normalization: str
    target_peak: float
    target_rms: float
    maximum_gain_db: float | None
    min_duration: float
    min_rms: float


def sha1_config(value: dict[str, Any]) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if args.hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if args.n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if args.bins_per_octave <= 0:
        raise ValueError("bins_per_octave must be positive")
    if args.fmin is None or args.fmin <= 0.0:
        raise ValueError("--fmin must be specified and positive")
    if not math.isfinite(args.db_min) or not math.isfinite(args.db_max):
        raise ValueError("db_min and db_max must be finite")
    if args.db_min >= args.db_max:
        raise ValueError("db_min must be less than db_max")
    if args.target_peak <= 0.0 or args.target_peak > 1.0:
        raise ValueError("target_peak must be in (0, 1]")
    if args.target_rms <= 0.0:
        raise ValueError("target_rms must be positive")
    if args.maximum_gain_db is not None and args.maximum_gain_db <= 0.0:
        raise ValueError("maximum_gain_db must be positive when supplied")
    if args.trim_margin_seconds < 0.0:
        raise ValueError("trim_margin_seconds must be non-negative")
    if args.frame_rms_length <= 0:
        raise ValueError("frame_rms_length must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if not 0.0 < args.global_reference_percentile <= 100.0:
        raise ValueError("global_reference_percentile must be in (0, 100]")

    highest_center = args.fmin * 2.0 ** ((args.n_bins - 1) / args.bins_per_octave)
    if highest_center > args.sample_rate / 2.0:
        raise ValueError(
            f"Highest CQT center {highest_center:.2f} Hz exceeds Nyquist "
            f"{args.sample_rate / 2.0:.2f} Hz"
        )


def trim_with_margin(
    y: np.ndarray,
    *,
    sample_rate: int,
    top_db: float,
    margin_seconds: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    if y.size == 0:
        return y.astype(np.float32), (0, 0)

    _, interval = librosa.effects.trim(y, top_db=top_db)
    if int(interval[1]) <= int(interval[0]):
        return y.astype(np.float32), (0, len(y))

    margin = int(round(margin_seconds * sample_rate))
    start = max(0, int(interval[0]) - margin)
    end = min(len(y), int(interval[1]) + margin)
    return y[start:end].astype(np.float32), (start, end)


def normalize_audio(
    y: np.ndarray,
    *,
    mode: str,
    target_peak: float,
    target_rms: float,
    maximum_gain_db: float | None,
) -> tuple[np.ndarray, float]:
    if y.size == 0 or mode == "none":
        return y.astype(np.float32), 1.0

    if mode == "peak":
        denominator = float(np.max(np.abs(y)))
        target = target_peak
    elif mode == "rms":
        denominator = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
        target = target_rms
    else:
        raise ValueError(f"Unsupported normalization mode: {mode!r}")

    if denominator < EPS:
        return y.astype(np.float32), 1.0

    gain = target / denominator
    if maximum_gain_db is not None:
        gain = min(gain, 10.0 ** (maximum_gain_db / 20.0))

    return (y * gain).astype(np.float32), float(gain)


def preprocess_audio(
    path: Path,
    *,
    cqt: CQTSettings,
    preprocessing: PreprocessSettings,
) -> tuple[np.ndarray, dict[str, Any]]:
    y, sr = load_audio(path, sample_rate=cqt.sample_rate, mono=True)
    raw = audio_stats(y, sr)

    trim_interval = (0, len(y))
    if preprocessing.trim_silence:
        y, trim_interval = trim_with_margin(
            y,
            sample_rate=sr,
            top_db=preprocessing.trim_top_db,
            margin_seconds=preprocessing.trim_margin_seconds,
        )

    y, gain = normalize_audio(
        y,
        mode=preprocessing.normalization,
        target_peak=preprocessing.target_peak,
        target_rms=preprocessing.target_rms,
        maximum_gain_db=preprocessing.maximum_gain_db,
    )

    processed = audio_stats(y, sr)
    metadata = {
        "raw_stats": raw,
        "processed_stats": processed,
        "trim_interval_samples": [int(trim_interval[0]), int(trim_interval[1])],
        "normalization_gain": gain,
        "normalization_gain_db": float(20.0 * math.log10(max(gain, EPS))),
    }
    return y, metadata


def compute_cqt(y: np.ndarray, settings: CQTSettings) -> np.ndarray:
    return librosa.cqt(
        y,
        sr=settings.sample_rate,
        hop_length=settings.hop_length,
        n_bins=settings.n_bins,
        bins_per_octave=settings.bins_per_octave,
        fmin=settings.fmin,
        filter_scale=settings.filter_scale,
        norm=settings.norm,
        sparsity=settings.sparsity,
        window=settings.window,
        scale=settings.scale,
        pad_mode=settings.pad_mode,
    )


def magnitude_to_normalized_db(
    magnitude: np.ndarray,
    *,
    reference_magnitude: float,
    db_min: float,
    db_max: float,
) -> np.ndarray:
    reference = max(float(reference_magnitude), EPS)
    db = librosa.amplitude_to_db(
        magnitude.astype(np.float32, copy=False),
        ref=reference,
        amin=EPS,
        top_db=None,
    )
    db = np.clip(db, db_min, db_max)
    normalized = 2.0 * ((db - db_min) / (db_max - db_min)) - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def compute_frame_rms(
    y: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    return librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
        pad_mode="constant",
    ).astype(np.float32)


def normalize_frame_rms(
    frame_rms: np.ndarray,
    *,
    floor_db: float,
) -> np.ndarray:
    if floor_db >= 0.0:
        raise ValueError("frame_rms_floor_db must be negative")

    db = 20.0 * np.log10(np.maximum(frame_rms, EPS))
    db = np.clip(db, floor_db, 0.0)
    normalized = 2.0 * ((db - floor_db) / -floor_db) - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def build_feature_config(
    args: argparse.Namespace,
    *,
    cqt: CQTSettings,
    preprocessing: PreprocessSettings,
    dataset_reference_magnitude: float | None,
) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "librosa_version": librosa.__version__,
        "numpy_version": np.__version__,
        "cqt": asdict(cqt),
        "preprocessing": asdict(preprocessing),
        "magnitude": {
            "db_min": args.db_min,
            "db_max": args.db_max,
            "amplitude_to_db_top_db": None,
            "reference_mode": args.reference_mode,
            "global_reference_percentile": args.global_reference_percentile,
            "dataset_reference_magnitude": dataset_reference_magnitude,
        },
        "storage": {
            "cache_dtype": args.cache_dtype,
            "store_phase": args.store_phase,
            "store_frame_rms": args.store_frame_rms,
            "frame_rms_length": args.frame_rms_length,
            "frame_rms_floor_db": args.frame_rms_floor_db,
            "store_normalized_frame_rms": args.store_normalized_frame_rms,
        },
    }


def load_existing_if_matching(path: Path, expected_hash: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return (
        isinstance(payload, dict)
        and payload.get("cache_version") == CACHE_VERSION
        and payload.get("config_hash") == expected_hash
    )


def calibrate_dataset_reference(
    files: list[Path],
    *,
    cqt: CQTSettings,
    preprocessing: PreprocessSettings,
    percentile: float,
    limit: int | None,
    workers: int,
) -> tuple[float, dict[str, Any]]:
    selected = files if limit is None else files[:limit]
    def calibrate_one(path: Path) -> float | None:
        try:
            y, metadata = preprocess_audio(path, cqt=cqt, preprocessing=preprocessing)
            stats = metadata["processed_stats"]
            if (
                stats["duration_seconds"] < preprocessing.min_duration
                or stats["rms"] < preprocessing.min_rms
            ):
                return None
            magnitude = np.abs(compute_cqt(y, cqt))
            return float(magnitude.max(initial=0.0))
        except Exception:
            return None

    maxima: list[float] = []
    skipped = 0

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cqt-ref") as executor:
        results = executor.map(calibrate_one, selected)
        for result in tqdm(
            results,
            total=len(selected),
            desc="Calibrating dataset CQT reference",
        ):
            if result is None:
                skipped += 1
            else:
                maxima.append(result)

    if not maxima:
        raise RuntimeError("No valid files were available for global-reference calibration")

    values = np.asarray(maxima, dtype=np.float64)
    reference = float(np.percentile(values, percentile))
    summary = {
        "number_of_files": int(values.size),
        "skipped_files": skipped,
        "percentile": percentile,
        "reference_magnitude": reference,
        "maxima_min": float(values.min()),
        "maxima_p50": float(np.percentile(values, 50.0)),
        "maxima_p90": float(np.percentile(values, 90.0)),
        "maxima_p99": float(np.percentile(values, 99.0)),
        "maxima_max": float(values.max()),
    }
    return reference, summary


def preprocess_file(
    path: Path,
    *,
    source_root: Path,
    output_root: Path,
    domain: str,
    cqt: CQTSettings,
    preprocessing: PreprocessSettings,
    args: argparse.Namespace,
    feature_config: dict[str, Any],
    config_hash: str,
    dataset_reference_magnitude: float | None,
) -> dict[str, Any]:
    file_id = stable_id(path, source_root)
    output_path = output_root / f"{file_id}.pt"
    base = {
        "id": file_id,
        "domain": domain,
        "source_path": str(path.resolve()),
        "cache_path": str(output_path.resolve()),
        "cache_version": CACHE_VERSION,
        "config_hash": config_hash,
    }

    if output_path.exists() and not args.rebuild:
        if load_existing_if_matching(output_path, config_hash):
            return {**base, "status": "skipped_matching"}
        return {
            **base,
            "status": "stale_existing",
            "error": "Existing payload has a different cache version or configuration hash. Use --rebuild or a new cache root.",
        }

    try:
        y, preprocessing_metadata = preprocess_audio(path, cqt=cqt, preprocessing=preprocessing)
        stats = preprocessing_metadata["processed_stats"]
        if stats["duration_seconds"] < preprocessing.min_duration:
            return {**base, "status": "too_short", "duration_seconds": stats["duration_seconds"]}
        if stats["rms"] < preprocessing.min_rms:
            return {**base, "status": "too_quiet", "rms": stats["rms"]}

        complex_cqt = compute_cqt(y, cqt)
        magnitude = np.abs(complex_cqt).astype(np.float32, copy=False)
        track_reference = max(float(magnitude.max(initial=0.0)), EPS)
        if args.reference_mode == "track":
            applied_reference = track_reference
        else:
            if dataset_reference_magnitude is None:
                raise RuntimeError("Dataset-global reference is missing")
            applied_reference = dataset_reference_magnitude

        normalized = magnitude_to_normalized_db(
            magnitude,
            reference_magnitude=applied_reference,
            db_min=args.db_min,
            db_max=args.db_max,
        )
        cqt_tensor = torch.from_numpy(normalized).unsqueeze(0)
        cqt_tensor = cqt_tensor.half() if args.cache_dtype == "float16" else cqt_tensor.float()

        frame_rms = compute_frame_rms(
            y,
            frame_length=args.frame_rms_length,
            hop_length=cqt.hop_length,
        )
        rms_frames = int(frame_rms.shape[-1])
        cqt_frames = int(cqt_tensor.shape[-1])
        if abs(rms_frames - cqt_frames) > 1:
            raise RuntimeError(
                f"Frame alignment mismatch: CQT={cqt_frames}, frame RMS={rms_frames}"
            )
        common_frames = min(rms_frames, cqt_frames)
        if cqt_frames != common_frames:
            cqt_tensor = cqt_tensor[..., :common_frames]
            complex_cqt = complex_cqt[..., :common_frames]
        frame_rms = frame_rms[..., :common_frames]

        metadata = {
            **base,
            "status": "ok",
            **preprocessing_metadata,
            "cqt_frames": common_frames,
            "cqt_shape": list(cqt_tensor.shape),
            "track_cqt_max_magnitude": track_reference,
            "applied_cqt_reference_magnitude": float(applied_reference),
            "cqt_reference_mode": args.reference_mode,
            "cqt_floor_fraction": float((cqt_tensor.float() <= -0.999).float().mean()),
            "frame_rms_statistics": {
                "mean": float(frame_rms.mean()),
                "min": float(frame_rms.min()),
                "max": float(frame_rms.max()),
                "below_1e_3_fraction": float(np.mean(frame_rms < 1e-3)),
                "below_3e_3_fraction": float(np.mean(frame_rms < 3e-3)),
                "below_1e_2_fraction": float(np.mean(frame_rms < 1e-2)),
            },
        }

        payload: dict[str, Any] = {
            "cache_version": CACHE_VERSION,
            "config_hash": config_hash,
            "feature_config": feature_config,
            "metadata": metadata,
            "cqt_mag": cqt_tensor,
        }
        if args.store_phase:
            phase = torch.from_numpy(np.angle(complex_cqt).astype(np.float32))
            payload["cqt_phase"] = phase.half() if args.cache_dtype == "float16" else phase
        if args.store_frame_rms:
            payload["frame_rms"] = torch.from_numpy(frame_rms).float()

        if args.store_normalized_frame_rms:
            normalized_frame_rms = normalize_frame_rms(
                frame_rms,
                floor_db=args.frame_rms_floor_db,
            )
            payload["frame_rms_normalized"] = torch.from_numpy(
                normalized_frame_rms
            ).to(dtype=cqt_tensor.dtype)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(output_path)

        return {
            **base,
            "status": "ok",
            "duration_seconds": stats["duration_seconds"],
            "rms": stats["rms"],
            "peak": stats["peak"],
            "normalization_gain": preprocessing_metadata["normalization_gain"],
            "cqt_frames": common_frames,
            "cqt_floor_fraction": metadata["cqt_floor_fraction"],
        }
    except Exception as exc:
        return {
            **base,
            "status": "error",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reproducible shared-domain CQT cache (version 4)."
    )
    parser.add_argument(
        "--domain",
        choices=["poly", "chip", "both"],
        default="both",
        help="Process one domain or both domains with one shared reference.",
    )
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--poly-source-root", type=Path, default=None)
    parser.add_argument("--chip-source-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, required=True)

    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-bins", type=int, default=116)
    parser.add_argument("--bins-per-octave", type=int, default=12)
    parser.add_argument("--fmin", type=float, required=True)
    parser.add_argument("--filter-scale", type=float, default=1.0)
    parser.add_argument("--norm", type=float, default=1.0)
    parser.add_argument("--sparsity", type=float, default=0.0)
    parser.add_argument("--window", type=str, default="hann")
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--pad-mode", type=str, default="constant")

    parser.add_argument("--db-min", type=float, default=-80.0)
    parser.add_argument("--db-max", type=float, default=0.0)
    parser.add_argument(
        "--reference-mode",
        choices=["track", "dataset"],
        default="dataset",
    )
    parser.add_argument("--global-reference-percentile", type=float, default=99.0)
    parser.add_argument("--global-reference-calibration-limit", type=int, default=None)
    parser.add_argument("--global-reference-file", type=Path, default=None)

    parser.add_argument("--trim-silence", action="store_true")
    parser.add_argument("--trim-top-db", type=float, default=60.0)
    parser.add_argument("--trim-margin-seconds", type=float, default=0.5)
    parser.add_argument("--normalization", choices=["none", "peak", "rms"], default="peak")
    parser.add_argument("--target-peak", type=float, default=0.95)
    parser.add_argument("--target-rms", type=float, default=0.1)
    parser.add_argument("--maximum-gain-db", type=float, default=None)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--min-rms", type=float, default=1e-4)

    parser.add_argument("--frame-rms-length", type=int, default=1024)
    parser.add_argument("--frame-rms-floor-db", type=float, default=-80.0)
    parser.add_argument(
        "--no-store-frame-rms",
        action="store_false",
        dest="store_frame_rms",
        help="Disable raw FP32 RMS-envelope storage.",
    )
    parser.add_argument(
        "--no-store-normalized-frame-rms",
        action="store_false",
        dest="store_normalized_frame_rms",
        help="Disable [-1,1] log-RMS conditioning storage.",
    )
    parser.set_defaults(store_frame_rms=True, store_normalized_frame_rms=True)

    parser.add_argument("--store-phase", action="store_true")
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 1) // 2)))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve_domains(args: argparse.Namespace) -> dict[str, Path]:
    if args.domain == "both":
        if args.poly_source_root is None or args.chip_source_root is None:
            raise ValueError(
                "--domain both requires --poly-source-root and --chip-source-root"
            )
        return {
            "poly": args.poly_source_root.resolve(),
            "chip": args.chip_source_root.resolve(),
        }

    root = args.source_root
    if root is None:
        root = args.poly_source_root if args.domain == "poly" else args.chip_source_root
    if root is None:
        raise ValueError(
            f"--domain {args.domain} requires --source-root or the matching domain root"
        )
    return {args.domain: root.resolve()}


def process_domain(
    *,
    domain: str,
    source_root: Path,
    files: list[Path],
    cache_root: Path,
    cqt: CQTSettings,
    preprocessing: PreprocessSettings,
    args: argparse.Namespace,
    feature_config: dict[str, Any],
    config_hash: str,
    dataset_reference: float | None,
) -> list[dict[str, Any]]:
    output_root = cache_root / domain
    output_root.mkdir(parents=True, exist_ok=True)

    worker = partial(
        preprocess_file,
        source_root=source_root,
        output_root=output_root,
        domain=domain,
        cqt=cqt,
        preprocessing=preprocessing,
        args=args,
        feature_config=feature_config,
        config_hash=config_hash,
        dataset_reference_magnitude=dataset_reference,
    )

    with ThreadPoolExecutor(
        max_workers=args.workers,
        thread_name_prefix=f"cqt-{domain}",
    ) as executor:
        records = list(
            tqdm(
                executor.map(worker, files),
                total=len(files),
                desc=f"Preprocessing {domain}",
            )
        )

    write_jsonl(output_root / "index.jsonl", records)
    write_csv(output_root / "index.csv", records)
    return records


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.frame_rms_floor_db >= 0.0:
        raise ValueError("frame_rms_floor_db must be negative")

    domains = resolve_domains(args)
    cache_root = args.cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    cqt = CQTSettings(
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        n_bins=args.n_bins,
        bins_per_octave=args.bins_per_octave,
        fmin=args.fmin,
        filter_scale=args.filter_scale,
        norm=args.norm,
        sparsity=args.sparsity,
        window=args.window,
        scale=not args.no_scale,
        pad_mode=args.pad_mode,
    )
    preprocessing = PreprocessSettings(
        trim_silence=args.trim_silence,
        trim_top_db=args.trim_top_db,
        trim_margin_seconds=args.trim_margin_seconds,
        normalization=args.normalization,
        target_peak=args.target_peak,
        target_rms=args.target_rms,
        maximum_gain_db=args.maximum_gain_db,
        min_duration=args.min_duration,
        min_rms=args.min_rms,
    )

    domain_files: dict[str, list[Path]] = {}
    joint_files: list[Path] = []
    for domain, source_root in domains.items():
        files = list_audio_files(source_root)
        if args.limit is not None:
            files = files[: args.limit]
        if not files:
            raise RuntimeError(f"No {domain} audio files found under {source_root}")
        domain_files[domain] = files
        joint_files.extend(files)

    dataset_reference: float | None = None
    calibration_summary: dict[str, Any] | None = None
    if args.reference_mode == "dataset":
        reference_file = args.global_reference_file or (cache_root / "shared_global_reference.json")
        if reference_file.exists() and not args.rebuild:
            calibration_summary = json.loads(reference_file.read_text(encoding="utf-8"))
            dataset_reference = float(calibration_summary["reference_magnitude"])
        else:
            dataset_reference, calibration_summary = calibrate_dataset_reference(
                joint_files,
                cqt=cqt,
                preprocessing=preprocessing,
                percentile=args.global_reference_percentile,
                limit=args.global_reference_calibration_limit,
                workers=args.workers,
            )
            calibration_summary["domains"] = {
                domain: len(files) for domain, files in domain_files.items()
            }
            reference_file.write_text(
                json.dumps(calibration_summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    feature_config = build_feature_config(
        args,
        cqt=cqt,
        preprocessing=preprocessing,
        dataset_reference_magnitude=dataset_reference,
    )
    feature_config["shared_domains"] = sorted(domains)
    feature_config["domain_file_counts"] = {
        domain: len(files) for domain, files in domain_files.items()
    }
    config_hash = sha1_config(feature_config)

    manifest = {
        "cache_version": CACHE_VERSION,
        "config_hash": config_hash,
        "feature_config": feature_config,
        "dataset_reference_calibration": calibration_summary,
    }
    (cache_root / "shared_cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Domains: {', '.join(domains)}")
    for domain, root in domains.items():
        print(f"  {domain}: {root} ({len(domain_files[domain])} files)")
    print(f"Cache root: {cache_root}")
    print(f"Config hash: {config_hash}")
    print(f"Reference mode: {args.reference_mode}")
    print(f"Workers: {args.workers}")
    print(f"Raw frame RMS stored: {args.store_frame_rms}")
    print(f"Normalized frame RMS stored: {args.store_normalized_frame_rms}")
    if dataset_reference is not None:
        print(f"Shared dataset reference magnitude: {dataset_reference:.8g}")

    all_records: list[dict[str, Any]] = []
    for domain, source_root in domains.items():
        all_records.extend(
            process_domain(
                domain=domain,
                source_root=source_root,
                files=domain_files[domain],
                cache_root=cache_root,
                cqt=cqt,
                preprocessing=preprocessing,
                args=args,
                feature_config=feature_config,
                config_hash=config_hash,
                dataset_reference=dataset_reference,
            )
        )

    write_jsonl(cache_root / "combined_index.jsonl", all_records)
    write_csv(cache_root / "combined_index.csv", all_records)

    counts: dict[str, int] = {}
    for record in all_records:
        status = str(record.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    print("Done. Combined status counts:")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
