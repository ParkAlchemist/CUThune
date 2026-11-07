from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from matplotlib import pyplot as plt

from shared.src.chiptune_shared.cqt.cache import discover_cache_records
from vocoder.src.nes_vocoder.data.dataset import load_processed_waveform, pad_1d


def normalize_path_string(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/").lower()


def find_cache_payload_for_source_path(
    chip_cache_root: Path,
    source_path: Path,
) -> tuple[dict[str, Any], Path]:
    records = discover_cache_records(chip_cache_root)

    wanted = normalize_path_string(source_path)

    # First try matching index metadata.
    for record in records:
        record_source = record.get("source_path")

        if record_source is None:
            continue

        if normalize_path_string(record_source) == wanted:
            cache_path = Path(record["cache_path"])
            payload = torch.load(cache_path, map_location="cpu")
            return payload, cache_path

    # Fallback: load payloads and inspect metadata.
    for record in records:
        cache_path = Path(record["cache_path"])
        payload = torch.load(cache_path, map_location="cpu")
        metadata = payload.get("metadata", {})
        record_source = metadata.get("source_path")

        if record_source is None:
            continue

        if normalize_path_string(record_source) == wanted:
            return payload, cache_path

    raise FileNotFoundError(
        "Could not find matching CQT cache payload for source_path:\n"
        f"{source_path}"
    )


def build_fixed_vocoder_batch(
    chip_cache_root: Path,
    source_path: Path,
    frame_start: int,
    segment_frames: int,
    sample_rate: int,
    hop_length: int,
    trim_top_db: float = 60.0,
    target_peak: float = 0.95,
) -> dict[str, Any]:
    payload, cache_path = find_cache_payload_for_source_path(
        chip_cache_root=chip_cache_root,
        source_path=source_path,
    )

    metadata = payload.get("metadata", {})
    cqt = payload["cqt_mag"].float()

    # [1, F, T] -> [F, T]
    if cqt.ndim == 3 and cqt.shape[0] == 1:
        cqt = cqt[0]

    if cqt.ndim != 2:
        raise ValueError(f"Expected CQT [F,T], got {tuple(cqt.shape)}")

    frame_end = frame_start + segment_frames

    cqt_segment = cqt[:, frame_start:frame_end]

    if cqt_segment.shape[-1] < segment_frames:
        cqt_segment = torch.nn.functional.pad(
            cqt_segment,
            (0, segment_frames - cqt_segment.shape[-1]),
            value=-1.0,
        )

    waveform = load_processed_waveform(
        source_path=source_path,
        sample_rate=sample_rate,
        trim_top_db=trim_top_db,
        target_peak=target_peak,
    )

    segment_samples = segment_frames * hop_length
    sample_start = frame_start * hop_length
    sample_end = sample_start + segment_samples

    audio_segment = waveform[sample_start:sample_end]
    audio_segment = pad_1d(audio_segment, segment_samples).unsqueeze(0)

    return {
        "cqt": cqt_segment.unsqueeze(0),          # [1, F, T]
        "audio": audio_segment.unsqueeze(0),     # [1, 1, samples]
        "cache_path": [str(cache_path)],
        "source_path": [metadata.get("source_path", str(source_path))],
        "frame_start": torch.tensor([frame_start], dtype=torch.long),
        "frame_end": torch.tensor([frame_end], dtype=torch.long),
        "sample_start": torch.tensor([sample_start], dtype=torch.long),
        "sample_end": torch.tensor([sample_end], dtype=torch.long),
    }


def tensor_to_audio_np(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()

    if x.ndim == 3:
        x = x[0, 0]
    elif x.ndim == 2:
        x = x[0]
    elif x.ndim == 1:
        pass
    else:
        raise ValueError(f"Unexpected audio tensor shape: {tuple(x.shape)}")

    y = x.numpy().astype(np.float32)
    return np.clip(y, -1.0, 1.0)


def tensor_to_cqt_np(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()

    if x.ndim == 4:
        x = x[0, 0]
    elif x.ndim == 3:
        if x.shape[0] == 1:
            x = x[0]
        else:
            raise ValueError(f"Unexpected 3D CQT tensor shape: {tuple(x.shape)}")
    elif x.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected CQT tensor shape: {tuple(x.shape)}")

    return x.numpy().astype(np.float32)


def normalize_db_to_unit(
    db: np.ndarray,
    db_min: float = -80.0,
    db_max: float = 0.0,
) -> np.ndarray:
    db = np.clip(db, db_min, db_max)
    return ((db - db_min) / (db_max - db_min) * 2.0 - 1.0).astype(np.float32)


def audio_to_cqt_norm(
    y: np.ndarray,
    sample_rate: int,
    hop_length: int,
    n_bins: int,
    bins_per_octave: int,
    fmin_hz: float,
) -> np.ndarray:
    cqt = librosa.cqt(
        y,
        sr=sample_rate,
        hop_length=hop_length,
        fmin=fmin_hz,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    )

    mag = np.abs(cqt)

    ref = float(np.max(mag)) if mag.size else 1.0
    if ref < 1e-8:
        ref = 1.0

    db = librosa.amplitude_to_db(mag, ref=ref)
    return normalize_db_to_unit(db)


def audio_to_stft_db(
    y: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann",
        center=True,
    )
    mag = np.abs(stft)
    db = librosa.amplitude_to_db(mag, ref=1.0)
    return np.clip(db, -80.0, 0.0).astype(np.float32)


def match_time_dims(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = min(a.shape[-1], b.shape[-1])
    return a[..., :t], b[..., :t]


def plot_matrix(
    ax,
    matrix: np.ndarray,
    title: str,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Time", fontsize=8)
    ax.set_ylabel("Freq", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)


def save_diagnostic_figure(
    path: Path,
    conditioning_cqt: np.ndarray,
    real_cqt: np.ndarray,
    fake_cqt: np.ndarray,
    real_stft: np.ndarray,
    fake_stft: np.ndarray,
    real_audio: np.ndarray,
    fake_audio: np.ndarray,
    sample_rate: int,
    title: str,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = create_diagnostic_figure(
        conditioning_cqt=conditioning_cqt,
        real_cqt=real_cqt,
        fake_cqt=fake_cqt,
        real_stft=real_stft,
        fake_stft=fake_stft,
        real_audio=real_audio,
        fake_audio=fake_audio,
        sample_rate=sample_rate,
        title=title,
    )

    fig.savefig(path, dpi=dpi)
    plt.close(fig)

def create_diagnostic_figure(
    *,
    conditioning_cqt: np.ndarray,
    real_cqt: np.ndarray,
    fake_cqt: np.ndarray,
    real_stft: np.ndarray,
    fake_stft: np.ndarray,
    real_audio: np.ndarray,
    fake_audio: np.ndarray,
    sample_rate: int,
    title: str,
):
    # Existing plotting body...
    cond_aligned, fake_cqt_aligned = match_time_dims(conditioning_cqt, fake_cqt)
    real_cqt_aligned, fake_cqt_aligned_2 = match_time_dims(real_cqt, fake_cqt)

    cqt_diff_cond_fake = np.abs(cond_aligned - fake_cqt_aligned)
    cqt_diff_real_fake = np.abs(real_cqt_aligned - fake_cqt_aligned_2)

    real_stft_aligned, fake_stft_aligned = match_time_dims(real_stft, fake_stft)
    stft_diff = np.abs(real_stft_aligned - fake_stft_aligned)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(16, 14),
        constrained_layout=True,
    )

    plot_matrix(
        axes[0, 0],
        conditioning_cqt,
        "Conditioning cached CQT",
        vmin=-1.0,
        vmax=1.0,
    )
    plot_matrix(
        axes[0, 1],
        real_cqt,
        "CQT extracted from real waveform",
        vmin=-1.0,
        vmax=1.0,
    )
    plot_matrix(
        axes[0, 2],
        fake_cqt,
        "CQT extracted from generated waveform",
        vmin=-1.0,
        vmax=1.0,
    )

    plot_matrix(
        axes[1, 0],
        cqt_diff_cond_fake,
        "|conditioning CQT - fake waveform CQT|",
        cmap="viridis",
        vmin=0.0,
        vmax=2.0,
    )
    plot_matrix(
        axes[1, 1],
        cqt_diff_real_fake,
        "|real waveform CQT - fake waveform CQT|",
        cmap="viridis",
        vmin=0.0,
        vmax=2.0,
    )
    axes[1, 2].axis("off")

    plot_matrix(
        axes[2, 0],
        real_stft,
        "Real waveform STFT dB",
        vmin=-80.0,
        vmax=0.0,
    )
    plot_matrix(
        axes[2, 1],
        fake_stft,
        "Generated waveform STFT dB",
        vmin=-80.0,
        vmax=0.0,
    )
    plot_matrix(
        axes[2, 2],
        stft_diff,
        "|real STFT - fake STFT|",
        cmap="viridis",
    )

    t_real = np.arange(len(real_audio)) / sample_rate
    t_fake = np.arange(len(fake_audio)) / sample_rate

    axes[3, 0].plot(t_real, real_audio, linewidth=0.7)
    axes[3, 0].set_title("Real waveform", fontsize=9)
    axes[3, 0].set_xlabel("Seconds")
    axes[3, 0].set_ylim(-1.05, 1.05)

    axes[3, 1].plot(t_fake, fake_audio, linewidth=0.7)
    axes[3, 1].set_title("Generated waveform", fontsize=9)
    axes[3, 1].set_xlabel("Seconds")
    axes[3, 1].set_ylim(-1.05, 1.05)

    min_len = min(len(real_audio), len(fake_audio))
    axes[3, 2].plot(t_real[:min_len], real_audio[:min_len], linewidth=0.7, label="real")
    axes[3, 2].plot(t_real[:min_len], fake_audio[:min_len], linewidth=0.7, alpha=0.7, label="fake")
    axes[3, 2].set_title("Overlay", fontsize=9)
    axes[3, 2].set_xlabel("Seconds")
    axes[3, 2].set_ylim(-1.05, 1.05)
    axes[3, 2].legend(fontsize=8)

    fig.suptitle(title, fontsize=13)

    return fig


