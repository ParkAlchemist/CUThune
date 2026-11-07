from __future__ import annotations

from pathlib import Path
import sys
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import librosa.display

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.src.chiptune_shared.cqt.features import (
    load_audio,
    trim_silence,
    peak_normalize,
    rms_normalize,
    audio_stats,
    compute_complex_cqt,
    complex_cqt_to_normalized_mag,
    compute_phase,
    compute_chroma_from_cqt_mag,
    reconstruct_with_phase,
    save_wav,
    cqt_frequency_range,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)

    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-bins", type=int, default=84)
    parser.add_argument("--bins-per-octave", type=int, default=12)
    parser.add_argument("--fmin", type=float, default=None)

    parser.add_argument("--db-min", type=float, default=-80.0)
    parser.add_argument("--db-max", type=float, default=0.0)

    parser.add_argument("--trim-silence", action="store_true")
    parser.add_argument("--trim-top-db", type=float, default=60.0)

    parser.add_argument(
        "--normalization",
        choices=["none", "peak", "rms"],
        default="peak",
    )
    parser.add_argument("--target-peak", type=float, default=0.95)
    parser.add_argument("--target-rms", type=float, default=0.1)

    parser.add_argument("--max-seconds", type=float, default=30.0)

    return parser.parse_args()


def save_cqt_plot(path: Path, cqt_mag_norm: np.ndarray, sample_rate: int, hop_length: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(
        cqt_mag_norm,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="cqt_note",
    )
    plt.colorbar(format="%.2f")
    plt.title("Normalized CQT magnitude [-1, 1]")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_chroma_plot(path: Path, chroma: np.ndarray, sample_rate: int, hop_length: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 4))
    librosa.display.specshow(
        chroma,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="chroma",
    )
    plt.colorbar(format="%.2f")
    plt.title("CQT-derived chroma")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def reconstruction_metrics(reference: np.ndarray, reconstructed: np.ndarray) -> dict:
    n = min(len(reference), len(reconstructed))
    ref = reference[:n]
    rec = reconstructed[:n]

    raw_error = ref - rec
    raw_mse = float(np.mean(raw_error ** 2))
    raw_mae = float(np.mean(np.abs(raw_error)))

    # Best scalar gain for reconstructed signal.
    denom = float(np.dot(rec, rec)) + 1e-12
    gain = float(np.dot(ref, rec) / denom)
    rec_scaled = rec * gain

    scaled_error = ref - rec_scaled
    scaled_mse = float(np.mean(scaled_error ** 2))
    scaled_mae = float(np.mean(np.abs(scaled_error)))

    return {
        "raw_mse": raw_mse,
        "raw_mae": raw_mae,
        "gain": gain,
        "scaled_mse": scaled_mse,
        "scaled_mae": scaled_mae,
    }



def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    y, sr = load_audio(args.audio, sample_rate=args.sample_rate, mono=True)

    if args.max_seconds is not None:
        max_samples = int(args.max_seconds * args.sample_rate)
        y = y[:max_samples]

    raw_stats = audio_stats(y, sr)

    if args.trim_silence:
        y = trim_silence(y, top_db=args.trim_top_db)

    if args.normalization == "peak":
        y = peak_normalize(y, target_peak=args.target_peak)
    elif args.normalization == "rms":
        y = rms_normalize(y, target_rms=args.target_rms)

    processed_stats = audio_stats(y, sr)

    cqt_complex = compute_complex_cqt(
        y,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        n_bins=args.n_bins,
        bins_per_octave=args.bins_per_octave,
        fmin=args.fmin,
    )

    y_direct = librosa.icqt(
        cqt_complex,
        sr = args.sample_rate,
        hop_length = args.hop_length,
        bins_per_octave = args.bins_per_octave,
        fmin = args.fmin,
        length = len(y),
    ).astype(np.float32)
    save_wav(args.out_dir / "reconstruction_direct_complex_cqt.wav", y_direct,
             args.sample_rate)

    cqt_mag, cqt_ref_mag = complex_cqt_to_normalized_mag(
        cqt_complex,
        db_min=args.db_min,
        db_max=args.db_max,
    )

    phase = compute_phase(cqt_complex)

    chroma = compute_chroma_from_cqt_mag(
        cqt_mag,
        bins_per_octave=args.bins_per_octave,
    )

    y_recon = reconstruct_with_phase(
        cqt_mag_norm=cqt_mag,
        phase=phase,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        bins_per_octave=args.bins_per_octave,
        db_min=args.db_min,
        db_max=args.db_max,
        ref_mag=cqt_ref_mag,
        fmin=args.fmin,
        length=len(y),
    )

    save_wav(args.out_dir / "preprocessed_input.wav", y, args.sample_rate)
    save_wav(args.out_dir / "reconstruction_mag_plus_original_phase.wav", y_recon, args.sample_rate)

    save_cqt_plot(
        args.out_dir / "cqt.png",
        cqt_mag,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
    )

    save_chroma_plot(
        args.out_dir / "chroma.png",
        chroma,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
    )

    freq_range = cqt_frequency_range(
        sample_rate=args.sample_rate,
        n_bins=args.n_bins,
        bins_per_octave=args.bins_per_octave,
        fmin=args.fmin,
    )

    phase_metrics = reconstruction_metrics(y, y_recon)
    direct_metrics = reconstruction_metrics(y, y_direct)

    mse = phase_metrics["raw_mse"]
    mae = phase_metrics["raw_mae"]

    report = {
        "audio": str(args.audio),
        "sample_rate": args.sample_rate,
        "hop_length": args.hop_length,
        "n_bins": args.n_bins,
        "bins_per_octave": args.bins_per_octave,
        "fmin": args.fmin,
        "db_min": args.db_min,
        "db_max": args.db_max,
        "raw_stats": raw_stats,
        "processed_stats": processed_stats,
        "cqt_shape": list(cqt_mag.shape),
        "cqt_ref_mag": cqt_ref_mag,
        "chroma_shape": list(chroma.shape),
        "reconstruction_mse": mse,
        "reconstruction_mae": mae,
        "reconstruction_mag_plus_phase_metrics": phase_metrics,
        "reconstruction_direct_complex_cqt_metrics": direct_metrics,
        "cqt_frequency_range": freq_range,
        "outputs": {
            "preprocessed_input": "preprocessed_input.wav",
            "reconstruction": "reconstruction_mag_plus_original_phase.wav",
            "cqt_plot": "cqt.png",
            "chroma_plot": "chroma.png",
        },
    }

    with (args.out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Smoke test complete.")
    print(f"CQT shape: {cqt_mag.shape}")
    print(f"Chroma shape: {chroma.shape}")
    print(f"Reconstruction MSE: {mse:.8f}")
    print(f"Reconstruction MAE: {mae:.8f}")
    print(f"Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
