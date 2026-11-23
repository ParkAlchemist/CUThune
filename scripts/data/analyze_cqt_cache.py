from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np
import torch
from tqdm import tqdm


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class CacheAuditRecord:
    cache_path: str
    source_path: str
    cache_version: int | None
    config_hash: str | None

    status: str
    error: str | None

    dtype: str | None
    channels: int | None
    cqt_bins: int | None
    cqt_frames: int | None

    cqt_min: float | None
    cqt_max: float | None
    cqt_mean: float | None
    cqt_std: float | None

    exact_negative_one_fraction: float | None
    below_negative_0999_fraction: float | None
    below_negative_099_fraction: float | None
    below_negative_095_fraction: float | None

    frame_floor_fraction_mean: float | None
    frame_floor_fraction_p50: float | None
    frame_floor_fraction_p90: float | None
    frame_floor_fraction_p99: float | None

    high_floor_frame_fraction_090: float | None
    high_floor_frame_fraction_099: float | None

    cqt_ref_mag: float | None

    metadata_sample_rate: int | None
    metadata_hop_length: int | None
    metadata_n_bins: int | None
    metadata_bins_per_octave: int | None
    metadata_fmin: float | None
    metadata_db_min: float | None
    metadata_db_max: float | None

    raw_peak: float | None
    raw_rms: float | None
    processed_peak: float | None
    processed_rms: float | None
    estimated_normalization_gain: float | None

    waveform_available: bool
    waveform_frame_count: int | None

    waveform_frame_rms_mean: float | None
    waveform_frame_rms_p01: float | None
    waveform_frame_rms_p10: float | None
    waveform_frame_rms_p50: float | None

    waveform_silence_fraction_1e3: float | None
    waveform_silence_fraction_3e3: float | None
    waveform_silence_fraction_1e2: float | None

    silent_cqt_mismatch_fraction_1e3: float | None
    silent_cqt_mismatch_fraction_3e3: float | None
    floor_but_active_fraction_1e2: float | None

    rms_floor_correlation: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit an existing CQT cache for metadata consistency, "
            "floor saturation, and waveform/CQT silence alignment."
        ),
    )

    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help=(
            "Cache domain directory containing .pt files, "
            "for example .../chip."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cache_audit"),
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=(
            "Optional replacement source root. If omitted, "
            "metadata source_path values are used."
        ),
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help=(
            "Optional index.jsonl. Defaults to "
            "<cache-root>/index.jsonl."
        ),
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
    )

    parser.add_argument(
        "--hop-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--expected-bins",
        type=int,
        default=116,
    )

    parser.add_argument(
        "--expected-cache-version",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--expected-config-hash",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--frame-length",
        type=int,
        default=1024,
        help=(
            "Waveform RMS analysis window. "
            "1024 samples is about 23.2 ms at 44.1 kHz."
        ),
    )

    parser.add_argument(
        "--cqt-floor-threshold",
        type=float,
        default=-0.999,
    )

    parser.add_argument(
        "--cqt-active-threshold",
        type=float,
        default=-0.90,
        help=(
            "A frame above this mean normalized CQT value is treated "
            "as carrying substantial cached energy."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--random-sample",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
    )

    parser.add_argument(
        "--skip-waveform-analysis",
        action="store_true",
    )

    parser.add_argument(
        "--recreate-preprocessing",
        action="store_true",
        help=(
            "Reapply metadata-described trimming and normalization "
            "before waveform/CQT alignment analysis."
        ),
    )

    return parser.parse_args()


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if not path.exists():
        return records

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} "
                    f"of {path}: {exc}"
                ) from exc

            if isinstance(value, dict):
                records.append(value)

    return records


def discover_cache_paths(
    *,
    cache_root: Path,
    index_path: Path,
) -> list[Path]:
    indexed_records = load_jsonl(
        index_path,
    )

    indexed_paths: list[Path] = []

    for record in indexed_records:
        if record.get("status") not in {
            "ok",
            "skipped_existing",
        }:
            continue

        raw_path = record.get(
            "cache_path"
        )

        if not raw_path:
            continue

        path = Path(raw_path)

        if not path.is_absolute():
            path = (
                PROJECT_ROOT / path
            ).resolve()

        if path.exists():
            indexed_paths.append(path)

    if indexed_paths:
        return sorted(set(indexed_paths))

    return sorted(
        path
        for path in cache_root.rglob(
            "*.pt"
        )
        if path.is_file()
    )


def safe_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    try:
        result = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(result):
        return None

    return result


def safe_int(
    value: Any,
) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return None


def nested_float(
    mapping: dict[str, Any],
    parent: str,
    key: str,
) -> float | None:
    nested = mapping.get(parent)

    if not isinstance(
        nested,
        dict,
    ):
        return None

    return safe_float(
        nested.get(key)
    )


def resolve_source_path(
    *,
    source_path: str,
    source_root: Path | None,
) -> Path | None:
    if not source_path:
        return None

    path = Path(source_path)

    if path.exists():
        return path

    if source_root is None:
        return None

    candidates = [
        source_root / path.name,
    ]

    parts = path.parts

    for depth in range(
        2,
        min(
            len(parts),
            6,
        )
        + 1,
    ):
        candidates.append(
            source_root.joinpath(
                *parts[-depth:]
            )
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(
        source_root.rglob(
            path.name
        )
    )

    if len(matches) == 1:
        return matches[0]

    return None


def trim_silence(
    waveform: np.ndarray,
    *,
    top_db: float,
) -> np.ndarray:
    if waveform.size == 0:
        return waveform.astype(
            np.float32,
        )

    trimmed, _ = (
        librosa.effects.trim(
            waveform,
            top_db=top_db,
        )
    )

    if trimmed.size == 0:
        return waveform.astype(
            np.float32,
        )

    return trimmed.astype(
        np.float32,
    )


def peak_normalize(
    waveform: np.ndarray,
    *,
    target_peak: float,
) -> np.ndarray:
    peak = np.max(np.abs(waveform)) if waveform.size else 0.0

    if peak < 1e-8:
        return waveform.astype(
            np.float32,
        )

    return (
        waveform
        / peak
        * target_peak
    ).astype(
        np.float32,
    )


def rms_normalize(
    waveform: np.ndarray,
    *,
    target_rms: float,
) -> np.ndarray:
    rms = float(
        np.sqrt(
            np.mean(
                waveform**2
            )
        )
    ) if waveform.size else 0.0

    if rms < 1e-8:
        return waveform.astype(
            np.float32,
        )

    return (
        waveform
        / rms
        * target_rms
    ).astype(
        np.float32,
    )


def recreate_preprocessing(
    waveform: np.ndarray,
    *,
    metadata: dict[str, Any],
) -> np.ndarray:
    result = waveform.astype(
        np.float32,
        copy=False,
    )

    trim_enabled = bool(
        metadata.get(
            "trim_silence",
            False,
        )
    )

    if trim_enabled:
        result = trim_silence(
            result,
            top_db=float(
                metadata.get(
                    "trim_top_db",
                    60.0,
                )
            ),
        )

    normalization = metadata.get(
        "normalization",
        "peak",
    )

    if normalization == "peak":
        result = peak_normalize(
            result,
            target_peak=float(
                metadata.get(
                    "target_peak",
                    0.95,
                )
            ),
        )

    elif normalization == "rms":
        result = rms_normalize(
            result,
            target_rms=float(
                metadata.get(
                    "target_rms",
                    0.1,
                )
            ),
        )

    elif normalization == "none":
        pass

    else:
        raise ValueError(
            "Unsupported metadata "
            f"normalization: {normalization!r}."
        )

    return result


def frame_rms(
    waveform: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
    center: bool = True,
) -> np.ndarray:
    values = librosa.feature.rms(
        y=waveform,
        frame_length=frame_length,
        hop_length=hop_length,
        center=center,
        pad_mode="constant",
    )

    return values.reshape(
        -1,
    ).astype(
        np.float32,
    )


def align_frame_vectors(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    count = min(
        len(left),
        len(right),
    )

    return (
        left[:count],
        right[:count],
    )


def safe_quantile(
    values: np.ndarray,
    quantile: float,
) -> float | None:
    if values.size == 0:
        return None

    result = float(
        np.quantile(
            values,
            quantile,
        )
    )

    if not math.isfinite(result):
        return None

    return result


def safe_correlation(
    left: np.ndarray,
    right: np.ndarray,
) -> float | None:
    left, right = align_frame_vectors(
        left,
        right,
    )

    if len(left) < 3:
        return None

    if float(np.std(left)) < 1e-12:
        return None

    if float(np.std(right)) < 1e-12:
        return None

    correlation = float(
        np.corrcoef(
            left,
            right,
        )[0, 1]
    )

    if not math.isfinite(
        correlation
    ):
        return None

    return correlation


def analyze_one_cache(
    *,
    cache_path: Path,
    args: argparse.Namespace,
) -> CacheAuditRecord:
    empty_values = {
        "dtype": None,
        "channels": None,
        "cqt_bins": None,
        "cqt_frames": None,
        "cqt_min": None,
        "cqt_max": None,
        "cqt_mean": None,
        "cqt_std": None,
        "exact_negative_one_fraction": None,
        "below_negative_0999_fraction": None,
        "below_negative_099_fraction": None,
        "below_negative_095_fraction": None,
        "frame_floor_fraction_mean": None,
        "frame_floor_fraction_p50": None,
        "frame_floor_fraction_p90": None,
        "frame_floor_fraction_p99": None,
        "high_floor_frame_fraction_090": None,
        "high_floor_frame_fraction_099": None,
        "cqt_ref_mag": None,
        "metadata_sample_rate": None,
        "metadata_hop_length": None,
        "metadata_n_bins": None,
        "metadata_bins_per_octave": None,
        "metadata_fmin": None,
        "metadata_db_min": None,
        "metadata_db_max": None,
        "raw_peak": None,
        "raw_rms": None,
        "processed_peak": None,
        "processed_rms": None,
        "estimated_normalization_gain": None,
        "waveform_available": False,
        "waveform_frame_count": None,
        "waveform_frame_rms_mean": None,
        "waveform_frame_rms_p01": None,
        "waveform_frame_rms_p10": None,
        "waveform_frame_rms_p50": None,
        "waveform_silence_fraction_1e3": None,
        "waveform_silence_fraction_3e3": None,
        "waveform_silence_fraction_1e2": None,
        "silent_cqt_mismatch_fraction_1e3": None,
        "silent_cqt_mismatch_fraction_3e3": None,
        "floor_but_active_fraction_1e2": None,
        "rms_floor_correlation": None,
    }

    try:
        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(
            payload,
            dict,
        ):
            raise TypeError(
                "Cache payload is not a dictionary."
            )

        metadata = payload.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

        source_path = str(
            metadata.get(
                "source_path",
                "",
            )
        )

        cqt = payload.get(
            "cqt_mag"
        )

        if not isinstance(
            cqt,
            torch.Tensor,
        ):
            raise TypeError(
                "Payload does not contain tensor cqt_mag."
            )

        if cqt.ndim == 2:
            cqt = cqt.unsqueeze(0)

        if cqt.ndim != 3:
            raise ValueError(
                "Expected cqt_mag [C,F,T] or [F,T], "
                f"got {tuple(cqt.shape)}."
            )

        cqt_float = (
            cqt.detach()
            .cpu()
            .float()
        )

        if not torch.isfinite(
            cqt_float
        ).all():
            raise ValueError(
                "CQT contains non-finite values."
            )

        floor_mask = (
            cqt_float
            <= args.cqt_floor_threshold
        )

        frame_floor_fraction = (
            floor_mask
            .float()
            .mean(
                dim=(0, 1),
            )
            .numpy()
        )

        frame_mean = (
            cqt_float
            .mean(
                dim=(0, 1),
            )
            .numpy()
        )

        raw_peak = nested_float(
            metadata,
            "raw_stats",
            "peak",
        )

        raw_rms = nested_float(
            metadata,
            "raw_stats",
            "rms",
        )

        processed_peak = nested_float(
            metadata,
            "processed_stats",
            "peak",
        )

        processed_rms = nested_float(
            metadata,
            "processed_stats",
            "rms",
        )

        if (
            raw_peak is not None
            and processed_peak is not None
            and raw_peak > 1e-12
        ):
            estimated_gain = (
                processed_peak
                / raw_peak
            )
        else:
            estimated_gain = None

        status_errors: list[str] = []

        cache_version = safe_int(
            payload.get(
                "cache_version"
            )
        )

        config_hash = payload.get(
            "config_hash"
        )

        if (
            cache_version
            != args.expected_cache_version
        ):
            status_errors.append(
                "cache_version_mismatch"
            )

        if (
            args.expected_config_hash
            is not None
            and config_hash
            != args.expected_config_hash
        ):
            status_errors.append(
                "config_hash_mismatch"
            )

        if (
            cqt_float.shape[-2]
            != args.expected_bins
        ):
            status_errors.append(
                "cqt_bin_count_mismatch"
            )

        metadata_bins = safe_int(
            metadata.get(
                "n_bins"
            )
        )

        if (
            metadata_bins is not None
            and metadata_bins
            != cqt_float.shape[-2]
        ):
            status_errors.append(
                "metadata_bin_count_mismatch"
            )

        waveform_metrics = {
            key: value
            for key, value
            in empty_values.items()
            if key.startswith(
                "waveform_"
            )
            or key.startswith(
                "silent_cqt_"
            )
            or key.startswith(
                "floor_but_"
            )
            or key
            == "rms_floor_correlation"
        }

        source = resolve_source_path(
            source_path=source_path,
            source_root=args.source_root,
        )

        waveform_available = (
            source is not None
            and not args.skip_waveform_analysis
        )

        if waveform_available:
            waveform, loaded_sr = (
                librosa.load(
                    source,
                    sr=args.sample_rate,
                    mono=True,
                )
            )

            waveform = waveform.astype(
                np.float32,
            )

            if args.recreate_preprocessing:
                waveform = recreate_preprocessing(
                    waveform,
                    metadata=metadata,
                )

            rms_values = frame_rms(
                waveform,
                frame_length=args.frame_length,
                hop_length=args.hop_length,
                center=True,
            )

            (
                aligned_rms,
                aligned_floor,
            ) = align_frame_vectors(
                rms_values,
                frame_floor_fraction,
            )

            (
                _,
                aligned_frame_mean,
            ) = align_frame_vectors(
                rms_values,
                frame_mean,
            )

            silent_1e3 = (
                aligned_rms < 1e-3
            )

            silent_3e3 = (
                aligned_rms < 3e-3
            )

            silent_1e2 = (
                aligned_rms < 1e-2
            )

            cqt_substantial = (
                aligned_frame_mean
                > args.cqt_active_threshold
            )

            cqt_nearly_floor = (
                aligned_floor >= 0.99
            )

            waveform_metrics = {
                "waveform_available": True,
                "waveform_frame_count": int(
                    len(rms_values)
                ),
                "waveform_frame_rms_mean": float(
                    aligned_rms.mean()
                ) if aligned_rms.size else None,
                "waveform_frame_rms_p01": safe_quantile(
                    aligned_rms,
                    0.01,
                ),
                "waveform_frame_rms_p10": safe_quantile(
                    aligned_rms,
                    0.10,
                ),
                "waveform_frame_rms_p50": safe_quantile(
                    aligned_rms,
                    0.50,
                ),
                "waveform_silence_fraction_1e3": float(
                    silent_1e3.mean()
                ) if silent_1e3.size else None,
                "waveform_silence_fraction_3e3": float(
                    silent_3e3.mean()
                ) if silent_3e3.size else None,
                "waveform_silence_fraction_1e2": float(
                    silent_1e2.mean()
                ) if silent_1e2.size else None,
                "silent_cqt_mismatch_fraction_1e3": float(
                    (
                        silent_1e3
                        & cqt_substantial
                    ).mean()
                ) if silent_1e3.size else None,
                "silent_cqt_mismatch_fraction_3e3": float(
                    (
                        silent_3e3
                        & cqt_substantial
                    ).mean()
                ) if silent_3e3.size else None,
                "floor_but_active_fraction_1e2": float(
                    (
                        np.logical_not(silent_1e2)
                        & cqt_nearly_floor
                    ).mean()
                ) if silent_1e2.size else None,
                "rms_floor_correlation": safe_correlation(
                    np.log10(
                        np.maximum(
                            aligned_rms,
                            1e-8,
                        )
                    ),
                    1.0 - aligned_floor,
                ),
            }

        status = (
            "ok"
            if not status_errors
            else ";".join(
                status_errors
            )
        )

        return CacheAuditRecord(
            cache_path=str(
                cache_path
            ),
            source_path=source_path,
            cache_version=cache_version,
            config_hash=(
                str(config_hash)
                if config_hash is not None
                else None
            ),
            status=status,
            error=None,
            dtype=str(cqt.dtype),
            channels=int(
                cqt_float.shape[0]
            ),
            cqt_bins=int(
                cqt_float.shape[-2]
            ),
            cqt_frames=int(
                cqt_float.shape[-1]
            ),
            cqt_min=float(
                cqt_float.min()
            ),
            cqt_max=float(
                cqt_float.max()
            ),
            cqt_mean=float(
                cqt_float.mean()
            ),
            cqt_std=float(
                cqt_float.std(
                    unbiased=False
                )
            ),
            exact_negative_one_fraction=float(
                (
                    cqt_float == -1.0
                )
                .float()
                .mean()
            ),
            below_negative_0999_fraction=float(
                (
                    cqt_float <= -0.999
                )
                .float()
                .mean()
            ),
            below_negative_099_fraction=float(
                (
                    cqt_float <= -0.99
                )
                .float()
                .mean()
            ),
            below_negative_095_fraction=float(
                (
                    cqt_float <= -0.95
                )
                .float()
                .mean()
            ),
            frame_floor_fraction_mean=float(
                frame_floor_fraction.mean()
            ),
            frame_floor_fraction_p50=safe_quantile(
                frame_floor_fraction,
                0.50,
            ),
            frame_floor_fraction_p90=safe_quantile(
                frame_floor_fraction,
                0.90,
            ),
            frame_floor_fraction_p99=safe_quantile(
                frame_floor_fraction,
                0.99,
            ),
            high_floor_frame_fraction_090=float(
                (
                    frame_floor_fraction
                    >= 0.90
                ).mean()
            ),
            high_floor_frame_fraction_099=float(
                (
                    frame_floor_fraction
                    >= 0.99
                ).mean()
            ),
            cqt_ref_mag=safe_float(
                metadata.get(
                    "cqt_ref_mag"
                )
            ),
            metadata_sample_rate=safe_int(
                metadata.get(
                    "sample_rate"
                )
            ),
            metadata_hop_length=safe_int(
                metadata.get(
                    "hop_length"
                )
            ),
            metadata_n_bins=metadata_bins,
            metadata_bins_per_octave=safe_int(
                metadata.get(
                    "bins_per_octave"
                )
            ),
            metadata_fmin=safe_float(
                metadata.get(
                    "fmin"
                )
            ),
            metadata_db_min=safe_float(
                metadata.get(
                    "db_min"
                )
            ),
            metadata_db_max=safe_float(
                metadata.get(
                    "db_max"
                )
            ),
            raw_peak=raw_peak,
            raw_rms=raw_rms,
            processed_peak=processed_peak,
            processed_rms=processed_rms,
            estimated_normalization_gain=(
                estimated_gain
            ),
            **waveform_metrics,
        )

    except Exception as exc:
        return CacheAuditRecord(
            cache_path=str(
                cache_path
            ),
            source_path="",
            cache_version=None,
            config_hash=None,
            status="error",
            error=repr(exc),
            **empty_values,
        )


def optional_values(
    records: Iterable[
        CacheAuditRecord
    ],
    field: str,
) -> np.ndarray:
    values: list[float] = []

    for record in records:
        value = getattr(
            record,
            field,
        )

        if value is None:
            continue

        value_float = float(value)

        if math.isfinite(
            value_float
        ):
            values.append(
                value_float
            )

    return np.asarray(
        values,
        dtype=np.float64,
    )


def summarize_field(
    records: list[
        CacheAuditRecord
    ],
    field: str,
) -> dict[str, float | int | None]:
    values = optional_values(
        records,
        field,
    )

    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p01": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "max": None,
        }

    return {
        "count": int(values.size),
        "mean": float(
            values.mean()
        ),
        "min": float(
            values.min()
        ),
        "p01": float(
            np.quantile(
                values,
                0.01,
            )
        ),
        "p10": float(
            np.quantile(
                values,
                0.10,
            )
        ),
        "p50": float(
            np.quantile(
                values,
                0.50,
            )
        ),
        "p90": float(
            np.quantile(
                values,
                0.90,
            )
        ),
        "p99": float(
            np.quantile(
                values,
                0.99,
            )
        ),
        "max": float(
            values.max()
        ),
    }


def top_records(
    records: list[
        CacheAuditRecord
    ],
    field: str,
    *,
    count: int = 25,
    descending: bool = True,
) -> list[dict[str, Any]]:
    candidates = [
        record
        for record in records
        if getattr(
            record,
            field,
        )
        is not None
    ]

    candidates.sort(
        key=lambda record: float(
            getattr(
                record,
                field,
            )
        ),
        reverse=descending,
    )

    result: list[
        dict[str, Any]
    ] = []

    for record in candidates[:count]:
        result.append({
            "field": field,
            "value": getattr(
                record,
                field,
            ),
            "cache_path": (
                record.cache_path
            ),
            "source_path": (
                record.source_path
            ),
            "cqt_frames": (
                record.cqt_frames
            ),
            "processed_rms": (
                record.processed_rms
            ),
            "processed_peak": (
                record.processed_peak
            ),
            "floor_fraction": (
                record
                .below_negative_0999_fraction
            ),
            "silence_fraction_3e3": (
                record
                .waveform_silence_fraction_3e3
            ),
            "silent_cqt_mismatch_3e3": (
                record
                .silent_cqt_mismatch_fraction_3e3
            ),
        })

    return result


def write_csv(
    path: Path,
    records: list[
        CacheAuditRecord
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [
        asdict(record)
        for record in records
    ]

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    args = parse_args()

    cache_root = (
        args.cache_root.resolve()
    )

    index_path = (
        args.index.resolve()
        if args.index is not None
        else cache_root
        / "index.jsonl"
    )

    output_dir = (
        args.output_dir.resolve()
    )

    cache_paths = discover_cache_paths(
        cache_root=cache_root,
        index_path=index_path,
    )

    if args.random_sample:
        random_generator = random.Random(
            args.seed
        )

        random_generator.shuffle(
            cache_paths
        )

    if args.limit is not None:
        cache_paths = cache_paths[
            :args.limit
        ]

    if not cache_paths:
        raise RuntimeError(
            f"No cache files found under {cache_root}."
        )

    print(
        f"Cache root:       {cache_root}"
    )

    print(
        f"Index:            {index_path}"
    )

    print(
        f"Cache files:      {len(cache_paths)}"
    )

    print(
        f"Waveform analysis: "
        f"{not args.skip_waveform_analysis}"
    )

    records = [
        analyze_one_cache(
            cache_path=path,
            args=args,
        )
        for path in tqdm(
            cache_paths,
            desc="Auditing cache",
        )
    ]

    status_counts = Counter(
        record.status
        for record in records
    )

    config_hash_counts = Counter(
        record.config_hash
        for record in records
        if record.config_hash
        is not None
    )

    dtype_counts = Counter(
        record.dtype
        for record in records
        if record.dtype is not None
    )

    summary_fields = [
        "cqt_frames",
        "cqt_mean",
        "cqt_std",
        "exact_negative_one_fraction",
        "below_negative_0999_fraction",
        "below_negative_099_fraction",
        "below_negative_095_fraction",
        "frame_floor_fraction_mean",
        "frame_floor_fraction_p90",
        "frame_floor_fraction_p99",
        "high_floor_frame_fraction_090",
        "high_floor_frame_fraction_099",
        "cqt_ref_mag",
        "raw_peak",
        "raw_rms",
        "processed_peak",
        "processed_rms",
        "estimated_normalization_gain",
        "waveform_frame_rms_mean",
        "waveform_silence_fraction_1e3",
        "waveform_silence_fraction_3e3",
        "waveform_silence_fraction_1e2",
        "silent_cqt_mismatch_fraction_1e3",
        "silent_cqt_mismatch_fraction_3e3",
        "floor_but_active_fraction_1e2",
        "rms_floor_correlation",
    ]

    summary = {
        "cache_root": str(
            cache_root
        ),
        "index_path": str(
            index_path
        ),
        "number_of_files": len(
            records
        ),
        "status_counts": dict(
            status_counts
        ),
        "config_hash_counts": dict(
            config_hash_counts
        ),
        "dtype_counts": dict(
            dtype_counts
        ),
        "field_summaries": {
            field: summarize_field(
                records,
                field,
            )
            for field
            in summary_fields
        },
        "worst_floor_occupancy": (
            top_records(
                records,
                "below_negative_0999_fraction",
            )
        ),
        "worst_silent_cqt_mismatch": (
            top_records(
                records,
                "silent_cqt_mismatch_fraction_3e3",
            )
        ),
        "largest_normalization_gains": (
            top_records(
                records,
                "estimated_normalization_gain",
            )
        ),
        "lowest_cqt_rms_correlations": (
            top_records(
                records,
                "rms_floor_correlation",
                descending=False,
            )
        ),
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "cache_audit_records.csv"
    )

    summary_path = (
        output_dir
        / "cache_audit_summary.json"
    )

    write_csv(
        csv_path,
        records,
    )

    write_json(
        summary_path,
        summary,
    )

    print("\nAudit complete.")

    print(
        f"Detailed CSV: {csv_path}"
    )

    print(
        f"Summary JSON: {summary_path}"
    )

    print("\nStatus counts:")

    for status, count in sorted(
        status_counts.items()
    ):
        print(
            f"  {status}: {count}"
        )

    print("\nConfiguration hashes:")

    for config_hash, count in (
        config_hash_counts.most_common()
    ):
        print(
            f"  {config_hash}: {count}"
        )

    for field in (
        "below_negative_0999_fraction",
        "waveform_silence_fraction_3e3",
        "silent_cqt_mismatch_fraction_3e3",
        "floor_but_active_fraction_1e2",
        "estimated_normalization_gain",
        "rms_floor_correlation",
    ):
        field_summary = summary[
            "field_summaries"
        ][field]

        print(
            f"\n{field}:"
        )

        for key in (
            "count",
            "mean",
            "p50",
            "p90",
            "p99",
            "max",
        ):
            print(
                f"  {key}: "
                f"{field_summary[key]}"
            )


if __name__ == "__main__":
    main()


