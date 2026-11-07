from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import json

import numpy as np
import soundfile as sf
import torch
from matplotlib import pyplot as plt

from src.eval.vocoder_diagnostics import create_diagnostic_figure, audio_to_cqt_norm, audio_to_stft_db


def tensor_to_audio_np(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()

    if x.ndim == 3:
        x = x[0, 0]
    elif x.ndim == 2:
        x = x[0]

    y = x.numpy()

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak * 0.95

    return y.astype(np.float32)


def save_metadata(
    path: Path,
    payload: dict,
) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


@torch.no_grad()
def export_preview_wavs(
    *,
    preview_dir: Path,
    generator: torch.nn.Module,
    preview_batches: Iterable[dict],
    device: torch.device,
    sample_rate: int,
    num_samples: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    writer = None,
    global_step: int | None = None,
    hop_length: int,
    cqt_bins: int,
    bins_per_octave: int,
    fmin: float,
) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)

    was_training = generator.training
    generator.eval()

    try:
        written = 0
        for batch in preview_batches:
            cqt = batch["cqt"].to(
                device,
                non_blocking=True,
            )
            real_audio = batch["audio"].to(
                device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(
                    use_amp
                    and device.type == "cuda"
                ),
            ):
                fake_audio = generator(cqt)

            batch_size = int(cqt.shape[0])
            source_paths = batch.get(
                "source_path",
                [""] * batch_size,
            )

            for item_index in range(batch_size):
                if written >= num_samples:
                    return

                real_np = tensor_to_audio_np(
                    real_audio[item_index:item_index + 1]
                )
                fake_np = tensor_to_audio_np(
                    fake_audio[item_index:item_index + 1]
                )

                prefix = f"sample_{written:04d}"

                sf.write(
                    preview_dir / f"{prefix}_real.wav",
                    real_np,
                    sample_rate,
                )
                sf.write(
                    preview_dir / f"{prefix}_fake.wav",
                    fake_np,
                    sample_rate,
                )

                save_metadata(
                    preview_dir / f"{prefix}_metadata.json",
                    {
                        "sample_index": written,
                        "source_path": source_paths[item_index],
                        "frame_start": int(
                            batch["frame_start"][item_index]
                        ),
                        "frame_end": int(
                            batch["frame_end"][item_index]
                        ),
                        "sample_start": int(
                            batch["sample_start"][item_index]
                        ),
                        "sample_end": int(
                            batch["sample_end"][item_index]
                        ),
                    },
                )

                figure = create_diagnostic_figure(
                    conditioning_cqt=cqt[item_index].detach().float().cpu().numpy(),
                    real_cqt=audio_to_cqt_norm(
                        y=real_np,
                        sample_rate=sample_rate,
                        hop_length=hop_length,
                        n_bins=cqt_bins,
                        bins_per_octave=bins_per_octave,
                        fmin_hz=fmin,
                    ),
                    fake_cqt=audio_to_cqt_norm(
                        y=fake_np,
                        sample_rate=sample_rate,
                        hop_length=hop_length,
                        n_bins=cqt_bins,
                        bins_per_octave=bins_per_octave,
                        fmin_hz=fmin,
                    ),
                    real_stft=audio_to_stft_db(
                        y=real_np,
                        sample_rate=sample_rate,
                        n_fft=1024,
                        hop_length=hop_length,
                    ),
                    fake_stft=audio_to_stft_db(
                        y=fake_np,
                        sample_rate=sample_rate,
                        n_fft=1024,
                        hop_length=hop_length,
                    ),
                    real_audio=real_np,
                    fake_audio=fake_np,
                    sample_rate=sample_rate,
                    title=f"Vocoder preview sample {written:04d}",
                )

                if (
                        writer is not None
                        and global_step is not None
                ):
                    tag = f"preview/sample_{written:04d}"

                    writer.add_figure(
                        f"{tag}/diagnostics",
                        figure,
                        global_step = global_step,
                        close = True,
                    )

                    writer.add_audio(
                        f"{tag}/real",
                        torch.from_numpy(
                            real_np
                        ).unsqueeze(0),
                        global_step = global_step,
                        sample_rate = sample_rate,
                    )

                    writer.add_audio(
                        f"{tag}/fake",
                        torch.from_numpy(
                            fake_np
                        ).unsqueeze(0),
                        global_step = global_step,
                        sample_rate = sample_rate,
                    )
                else:
                    plt.close(figure) \


                written += 1

    finally:
        generator.train(was_training)

