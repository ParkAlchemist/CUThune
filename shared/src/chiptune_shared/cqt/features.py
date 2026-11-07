from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch


AUDIO_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".aiff",
    ".aif",
)


def list_audio_files(root: Path, extensions: tuple[str, ...] = AUDIO_EXTENSIONS) -> list[Path]:
    files: list[Path] = []

    for ext in extensions:
        files.extend(root.rglob(f"*{ext}"))
        files.extend(root.rglob(f"*{ext.upper()}"))

    return sorted(set(files))


def stable_id(path: Path, root: Path | None = None) -> str:
    path = Path(path)
    key = str(path.relative_to(root)) if root else str(path)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def config_hash(config: dict[str, Any]) -> str:
    text = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def load_audio(
    path: Path,
    sample_rate: int,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=sample_rate, mono=mono)
    y = y.astype(np.float32)
    return y, int(sr)


def trim_silence(
    y: np.ndarray,
    top_db: float = 60.0,
) -> np.ndarray:
    if y.size == 0:
        return y

    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)

    if y_trimmed.size == 0:
        return y

    return y_trimmed.astype(np.float32)


def peak_normalize(
    y: np.ndarray,
    target_peak: float = 0.95,
) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0

    if peak < 1e-8:
        return y.astype(np.float32)

    return (y / peak * target_peak).astype(np.float32)


def rms_normalize(
    y: np.ndarray,
    target_rms: float = 0.1,
) -> np.ndarray:
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0

    if rms < 1e-8:
        return y.astype(np.float32)

    return (y / rms * target_rms).astype(np.float32)


def audio_stats(y: np.ndarray, sample_rate: int) -> dict[str, Any]:
    if y.size == 0:
        return {
            "duration_seconds": 0.0,
            "num_samples": 0,
            "sample_rate": sample_rate,
            "rms": 0.0,
            "peak": 0.0,
            "mean_abs": 0.0,
            "zero_fraction": 1.0,
        }

    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))
    mean_abs = float(np.mean(np.abs(y)))
    zero_fraction = float(np.mean(np.abs(y) < 1e-6))

    return {
        "duration_seconds": float(len(y) / sample_rate),
        "num_samples": int(len(y)),
        "sample_rate": int(sample_rate),
        "rms": rms,
        "peak": peak,
        "mean_abs": mean_abs,
        "zero_fraction": zero_fraction,
    }


def compute_complex_cqt(
    y: np.ndarray,
    sample_rate: int,
    hop_length: int,
    n_bins: int,
    bins_per_octave: int,
    fmin: float | None = None,
    filter_scale: float = 1.0,
    norm: float = 1,
    sparsity: float = 0.0,
    window: str = "hann",
    scale: bool = True,
) -> np.ndarray:
    cqt = librosa.cqt(
        y,
        sr=sample_rate,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
        filter_scale=filter_scale,
        norm=norm,
        sparsity=sparsity,
        window=window,
        scale=scale,
    )
    return cqt


def complex_cqt_to_normalized_mag(
    cqt_complex: np.ndarray,
    db_min: float,
    db_max: float,
) -> tuple[np.ndarray, float]:
    mag = np.abs(cqt_complex)

    ref_mag = float(np.max(mag))
    if ref_mag < 1e-8:
        ref_mag = 1.0

    db = librosa.amplitude_to_db(mag, ref=ref_mag)
    db = np.clip(db, db_min, db_max)

    norm = 2.0 * ((db - db_min) / (db_max - db_min)) - 1.0
    norm = np.clip(norm, -1.0, 1.0).astype(np.float32)

    return norm, ref_mag


def normalized_mag_to_amplitude(
    cqt_mag_norm: np.ndarray | torch.Tensor,
    db_min: float,
    db_max: float,
    ref_mag: float = 1.0
) -> np.ndarray:
    if isinstance(cqt_mag_norm, torch.Tensor):
        x = cqt_mag_norm.detach().cpu().float().numpy()
    else:
        x = cqt_mag_norm.astype(np.float32)

    db = ((x + 1.0) / 2.0) * (db_max - db_min) + db_min
    amp = librosa.db_to_amplitude(db, ref=ref_mag)

    return amp.astype(np.float32)


def compute_phase(cqt_complex: np.ndarray) -> np.ndarray:
    return np.angle(cqt_complex).astype(np.float32)


def compute_chroma_from_cqt_mag(
    cqt_mag_norm: np.ndarray,
    bins_per_octave: int = 12,
) -> np.ndarray:
    """
    Simple CQT-derived chroma by folding octaves.
    Input shape: [n_bins, frames]
    Output shape: [12, frames]
    """
    n_bins, frames = cqt_mag_norm.shape

    usable_bins = (n_bins // bins_per_octave) * bins_per_octave
    x = cqt_mag_norm[:usable_bins, :]

    x = x.reshape(-1, bins_per_octave, frames)
    chroma = x.mean(axis=0)

    return chroma.astype(np.float32)


def cqt_frequency_range(
    sample_rate: int,
    n_bins: int,
    bins_per_octave: int,
    fmin: float | None = None,
) -> dict[str, float]:
    if fmin is None:
        fmin = float(librosa.note_to_hz("C1"))

    fmax_center = float(fmin * (2.0 ** ((n_bins - 1) / bins_per_octave)))
    nyquist = float(sample_rate / 2.0)

    return {
        "fmin": float(fmin),
        "fmax_center": fmax_center,
        "nyquist": nyquist,
        "coverage_ratio_to_nyquist": fmax_center / nyquist,
    }


def reconstruct_with_phase(
    cqt_mag_norm: np.ndarray | torch.Tensor,
    phase: np.ndarray | torch.Tensor,
    sample_rate: int,
    hop_length: int,
    bins_per_octave: int,
    db_min: float,
    db_max: float,
    ref_mag: float = 1.0,
    fmin: float | None = None,
    length: int | None = None,
) -> np.ndarray:
    amp = normalized_mag_to_amplitude(
        cqt_mag_norm,
        db_min=db_min,
        db_max=db_max,
        ref_mag=ref_mag,
    )

    if isinstance(phase, torch.Tensor):
        phase_np = phase.detach().cpu().float().numpy()
    else:
        phase_np = phase.astype(np.float32)

    complex_cqt = amp * np.exp(1j * phase_np)

    y = librosa.icqt(
        complex_cqt,
        sr=sample_rate,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
        filter_scale=1.0,
        norm=1,
        window="hann",
        scale=True,
        length=length,
    )

    y = y.astype(np.float32)

    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 1.0:
        y = y / peak * 0.99

    return y


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_wav(path: Path, y: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, sample_rate)
