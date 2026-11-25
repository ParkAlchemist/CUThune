from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from vocoder.src.nes_vocoder.training.context import TrainingRuntime


@torch.no_grad()
def run_validation(
    *,
    runtime: TrainingRuntime,
    state,
    max_batches: int,
) -> dict[str, float]:
    generator = runtime.components.models.generator
    discriminator = runtime.components.models.discriminator

    generator_was_training = generator.training
    discriminator_was_training = discriminator.training
    generator.eval()
    discriminator.eval()

    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    try:
        for batch_index, batch in enumerate(runtime.validation_loader):
            if batch_index >= max_batches:
                break

            cqt = batch["cqt"].to(runtime.device, non_blocking=True)
            rms_envelope = batch["rms_envelope"].to(
                runtime.device,
                non_blocking=True,
            )
            real_audio = batch["audio"].to(
                runtime.device,
                non_blocking=True,
            )

            dtype = runtime.config.amp.dtype
            use_amp = runtime.config.amp.enabled
            device = runtime.device.type
            autocast_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
            autocast_enabled = (use_amp and device == "cuda")

            with torch.amp.autocast(
                device_type=runtime.device.type,
                dtype=autocast_dtype,
                enabled=(
                    autocast_enabled
                ),
            ):
                output = generator(
                    cqt,
                    rms_envelope,
                    return_diagnostics=True,
                )

            fake_audio = output.audio

            batch_metrics = {
                "validation/output_rms": float(
                    torch.sqrt(fake_audio.float().square().mean() + 1e-8)
                    .cpu()
                ),
                "validation/output_mean": float(
                    fake_audio.float().mean().cpu()
                ),
                "validation/negative_rail": float(
                    fake_audio.float().le(-0.99).float().mean().cpu()
                ),
                "validation/positive_rail": float(
                    fake_audio.float().ge(0.99).float().mean().cpu()
                ),
                "validation/l1": float(
                    torch.nn.functional.l1_loss(
                        fake_audio.float(),
                        real_audio.float(),
                    ).cpu()
                ),
            }

            if output.rms_embedding is not None:
                batch_metrics["validation/rms_to_cqt_ratio"] = float(
                    output.rms_embedding.to_cqt_ratio.cpu()
                )

            for name, value in batch_metrics.items():
                sums[name] += value
                counts[name] += 1

        means = {
            name: total / counts[name]
            for name, total in sums.items()
            if counts[name] > 0
        }

        if runtime.writer is not None:
            for name, value in means.items():
                runtime.writer.add_scalar(name, value, state.global_step)

        return means
    finally:
        generator.train(generator_was_training)
        discriminator.train(discriminator_was_training)


