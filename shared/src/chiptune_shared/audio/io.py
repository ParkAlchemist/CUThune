from pathlib import Path
import hashlib
import json
import random

import librosa
import numpy as np
import torch


def list_audio_files(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []

    for ext in extensions:
        files.extend(root.rglob(f"*{ext}"))
        files.extend(root.rglob(f"*{ext.upper()}"))

    return sorted(set(files))


def stable_file_id(path: Path, root: Path | None = None) -> str:
    """
    Creates a stable id for cache filenames.
    Uses relative path if root is given.
    """
    path = Path(path)
    key = str(path.relative_to(root)) if root else str(path)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def load_audio(
    path: Path,
    sample_rate: int,
    mono: bool = True,
    normalize_peak: bool = True,
) -> np.ndarray:
    y, _ = librosa.load(path, sr=sample_rate, mono=mono)

    if normalize_peak:
        peak = np.max(np.abs(y))
        if peak > 1e-8:
            y = y / peak

    return y.astype(np.float32)


def pad_or_crop(
    y: np.ndarray,
    target_samples: int,
    random_crop: bool = True,
) -> np.ndarray:
    if len(y) == target_samples:
        return y

    if len(y) < target_samples:
        pad = target_samples - len(y)
        return np.pad(y, (0, pad), mode="constant")

    max_start = len(y) - target_samples
    start = random.randint(0, max_start) if random_crop else 0
    return y[start : start + target_samples]


def audio_to_cqt_tensor(
    y: np.ndarray,
    sample_rate: int,
    hop_length: int,
    n_bins: int,
    bins_per_octave: int,
    db_min: float = -80.0,
    db_max: float = 0.0,
    fmin: float | None = None,
) -> torch.Tensor:
    """
    Returns tensor shape [1, n_bins, time_frames], normalized to [-1, 1].
    """
    cqt = librosa.cqt(
        y,
        sr=sample_rate,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
    )

    mag = np.abs(cqt)
    db = librosa.amplitude_to_db(mag, ref=np.max)

    db = np.clip(db, db_min, db_max)

    # Map [db_min, db_max] -> [-1, 1]
    norm = 2.0 * ((db - db_min) / (db_max - db_min)) - 1.0
    norm = np.clip(norm, -1.0, 1.0).astype(np.float32)

    return torch.from_numpy(norm).unsqueeze(0)


def cqt_tensor_to_db(
    cqt_tensor: torch.Tensor,
    db_min: float = -80.0,
    db_max: float = 0.0,
) -> torch.Tensor:
    """
    Converts normalized [-1, 1] CQT tensor back to dB scale.
    """
    return ((cqt_tensor + 1.0) / 2.0) * (db_max - db_min) + db_min


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records
    