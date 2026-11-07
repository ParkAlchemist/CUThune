import numpy as np
import librosa


def validate_cqt_configuration(
    sample_rate: int,
    hop_length: int,
    fmin_hz: float,
    n_bins: int,
    bins_per_octave: int,
    filter_scale: float = 1.0,
    test_seconds: float = 10.0,
) -> None:
    num_octaves = int(np.ceil(n_bins / bins_per_octave))
    frames_per_second = sample_rate / hop_length

    print("Validating CQT configuration:")
    print(f"  sample_rate:       {sample_rate}")
    print(f"  hop_length:        {hop_length}")
    print(f"  frames/sec:        {frames_per_second:.3f}")
    print(f"  fmin_hz:           {fmin_hz:.4f}")
    print(f"  n_bins:            {n_bins}")
    print(f"  bins_per_octave:   {bins_per_octave}")
    print(f"  octaves:           {num_octaves}")
    print(f"  filter_scale:      {filter_scale}")
    print(f"  test_seconds:      {test_seconds}")

    num_samples = int(round(sample_rate * test_seconds))

    # Low-amplitude noise exercises the transform more realistically than
    # an entirely silent signal.
    rng = np.random.default_rng(1337)
    test_audio = (
        rng.standard_normal(num_samples).astype(np.float32) * 1e-3
    )

    try:
        result = librosa.cqt(
            test_audio,
            sr=sample_rate,
            hop_length=hop_length,
            fmin=fmin_hz,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            filter_scale=filter_scale,
            tuning=0.0,
            pad_mode="constant",
        )
    except Exception as exc:
        raise ValueError(
            "Invalid CQT preprocessing configuration:\n"
            f"  sample_rate={sample_rate}\n"
            f"  hop_length={hop_length}\n"
            f"  fmin_hz={fmin_hz:.4f}\n"
            f"  n_bins={n_bins}\n"
            f"  bins_per_octave={bins_per_octave}\n"
            f"  num_octaves={num_octaves}\n"
            f"  filter_scale={filter_scale}\n"
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc

    expected_frames = 1 + num_samples // hop_length

    print("\nCQT configuration validation passed:")
    print(f"  input samples:     {num_samples}")
    print(f"  output shape:      {result.shape}")
    print(f"  expected frames:   approximately {expected_frames}")
    print(f"  finite values:     {np.isfinite(result).all()}")

    if result.shape[0] != n_bins:
        raise AssertionError(
            f"Expected {n_bins} CQT bins, got {result.shape[0]}"
        )

    if not np.isfinite(result).all():
        raise AssertionError("CQT contains NaN or infinite values.")



if __name__ == "__main__":
    validate_cqt_configuration(
        sample_rate=44100,
        hop_length=256,
        fmin_hz=float(librosa.note_to_hz("C1")),
        n_bins=202,
        bins_per_octave=24,
        filter_scale=1.0,
        test_seconds=10.0,
    )

    validate_cqt_configuration(
        sample_rate=44100,
        hop_length=256,
        fmin_hz=float(librosa.note_to_hz("A0")),
        n_bins=116,
        bins_per_octave=12,
        filter_scale=1.0,
        test_seconds=10.0,
    )

