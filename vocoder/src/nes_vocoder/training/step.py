from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.losses.vocoder_losses import (
    VocoderLossBundle,
    compute_vocoder_discriminator_loss,
    compute_vocoder_generator_loss,
    detach_vocoder_loss_dict, discriminator_family_losses, family_balanced_discriminator_loss,
)
from src.losses.vocoder.metrics import generator_loss_metrics, discriminator_loss_metrics
from src.losses.vocoder.adversarial_loss import generator_family_adversarial_losses
from src.losses.vocoder.feature_matching_loss import generator_family_feature_matching_losses
from src.training.vocoder.logging import (
    discriminator_branch_metrics,
    discriminator_health_metrics,
)


@dataclass
class VocoderModels:
    generator: nn.Module
    discriminator: nn.Module


@dataclass
class VocoderOptimizers:
    generator: torch.optim.Optimizer
    discriminator: torch.optim.Optimizer


@dataclass
class VocoderMicroStepResult:
    losses: dict[str, float]
    batch_size: int


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = requires_grad


def zero_vocoder_gradients(
    optimizers: VocoderOptimizers,
) -> None:
    optimizers.generator.zero_grad(
        set_to_none=True
    )

    optimizers.discriminator.zero_grad(
        set_to_none=True
    )


def vocoder_train_micro_step(
        *,
        batch: dict,
        models: VocoderModels,
        loss_bundle: VocoderLossBundle,
        device: torch.device,
        use_amp: bool = False,
        scaler: torch.amp.GradScaler | None = None,
        dtype: str,
        loss_divisor: int,
) -> VocoderMicroStepResult:
    if loss_divisor < 1:
        raise ValueError(f"loss_divisor must be >= 1, got {loss_divisor}")

    generator = models.generator
    discriminator = models.discriminator

    cqt = batch["cqt"].to(device, non_blocking=True)
    real_audio = batch["audio"].to(device, non_blocking=True)

    autocast_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
    autocast_enabled = (use_amp and device.type == "cuda")

    #--------------------------------------------------------
    # Discriminator Backwards Pass
    #--------------------------------------------------------
    with torch.amp.autocast(
        device_type=device.type,
        enabled=autocast_enabled,
        dtype=autocast_dtype,
    ):
        with torch.no_grad():
            fake_audio_for_d = generator(cqt)

        discriminator_outputs_d = discriminator(
            real_audio,
            fake=fake_audio_for_d.detach(),
        )

        discriminator_losses = family_balanced_discriminator_loss(discriminator_outputs_d)

        health_metrics = discriminator_health_metrics(
            discriminator_outputs_d
        )

        health_metrics.update(
            discriminator_branch_metrics(
                discriminator_outputs_d,
                mpd_periods=tuple(
                    discriminator.config.mpd.periods,
                ),
                mrd_resolutions=tuple(
                    discriminator.config.mrd.resolutions
                ),
            )
        )

        discriminator_family = (
            discriminator_family_losses(
                discriminator_outputs_d
            )
        )

        loss_d_total = discriminator_losses.total
        loss_d_backward = loss_d_total / loss_divisor

    if scaler is not None and use_amp:
        scaler.scale(loss_d_backward).backward()
    else:
        loss_d_backward.backward()

    #-------------------------------------------------------------
    # Generator Backwards Pass
    #-------------------------------------------------------------
    set_requires_grad(discriminator, False)

    try:
        with torch.amp.autocast(
            device_type=device.type,
            enabled=autocast_enabled,
            dtype=autocast_dtype,
        ):
            fake_audio_for_g = generator(cqt)

            discriminator_outputs_g = discriminator(
                real_audio,
                fake=fake_audio_for_g,
            )

            generator_losses = compute_vocoder_generator_loss(
                discriminator_outputs=discriminator_outputs_g,
                fake_audio=fake_audio_for_g,
                real_audio=real_audio,
                loss_bundle=loss_bundle,
            )

        generator_backward_loss = generator_losses.total / loss_divisor

        if scaler is not None and use_amp:
            scaler.scale(generator_backward_loss).backward()
        else:
            generator_backward_loss.backward()
    finally:
        set_requires_grad(discriminator, True)

    losses = generator_loss_metrics(generator_losses)

    losses.update(discriminator_loss_metrics(discriminator_losses))

    losses.update(health_metrics)

    losses.update(
        discriminator_branch_metrics(
            discriminator_outputs_d,
            mpd_periods=tuple(
                models.discriminator.config.mpd.periods
            ),
            mrd_resolutions=tuple(
                tuple(resolution) for resolution in models.discriminator.config.mrd.resolutions
            ),
        )
    )

    fake_detached = (
        fake_audio_for_g.detach().float()
    )

    losses.update(
        {
            "generator_output/rms": float(
                fake_detached
                .square()
                .mean()
                .sqrt()
                .cpu()
            ),
            "generator_output/peak": float(
                fake_detached
                .abs()
                .max()
                .cpu()
            ),
            "generator_output/mean": float(
                fake_detached.mean().cpu()
            ),
            "generator_output/std": float(
                fake_detached.std(
                    unbiased=False
                ).cpu()
            ),
            "generator_output/tanh_saturation_fraction": float(
                (
                        fake_detached.abs() >= 0.99
                )
                .float()
                .mean()
                .cpu()
            ),
        }
    )

    return VocoderMicroStepResult(losses=losses, batch_size=int(cqt.shape[0]))


def finish_vocoder_optimizer_step(
        *,
        models: VocoderModels,
        optimizers: VocoderOptimizers,
        scaler: torch.amp.GradScaler | None = None,
        use_amp: bool = False,
        grad_clip_generator: float | None = None,
        grad_clip_discriminator: float | None = None,
) -> dict[str, float | None]:

    generator_grad_norm: float = 0.0
    discriminator_grad_norm: float = 0.0

    if scaler is not None and use_amp:
        scaler.unscale_(optimizers.generator)
        scaler.unscale_(optimizers.discriminator)

    if grad_clip_generator is not None:
        generator_grad_norm_tensor = (
            torch.nn.utils.clip_grad_norm_(
                models.generator.parameters(),
                max_norm=grad_clip_generator,
            )
        )

        generator_grad_norm = float(
            generator_grad_norm_tensor.detach().cpu()
        )

    if grad_clip_discriminator is not None:
        discriminator_grad_norm_tensor = (
            torch.nn.utils.clip_grad_norm_(
                models.discriminator.parameters(),
                max_norm=grad_clip_discriminator,
            )
        )

        discriminator_grad_norm = float(
            discriminator_grad_norm_tensor.detach().cpu()
        )

    if scaler is not None and use_amp:
        scaler.step(optimizers.discriminator)
        scaler.step(optimizers.generator)
        scaler.update()
    else:
        optimizers.discriminator.step()
        optimizers.generator.step()

    optimizers.discriminator.zero_grad(set_to_none=True)
    optimizers.generator.zero_grad(set_to_none=True)

    return {
        "grad_norm_generator": float(generator_grad_norm),
        "grad_norm_discriminator": float(discriminator_grad_norm),
    }


def vocoder_train_step(
    *,
    batch: dict,
    models: VocoderModels,
    optimizers: VocoderOptimizers,
    loss_bundle: VocoderLossBundle,
    device: torch.device,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
    dtype: str,
    grad_clip_generator: float | None,
    grad_clip_discriminator: float | None,
) -> VocoderMicroStepResult:
    zero_vocoder_gradients(optimizers)

    micro_result = vocoder_train_micro_step(
        batch=batch,
        models=models,
        loss_bundle=loss_bundle,
        device=device,
        use_amp=use_amp,
        scaler=scaler,
        dtype=dtype,
        loss_divisor=1,
    )

    gradient_metrics = (
        finish_vocoder_optimizer_step(
            models=models,
            optimizers=optimizers,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip_generator=(
                grad_clip_generator
            ),
            grad_clip_discriminator=(
                grad_clip_discriminator
            ),
        )
    )

    losses = dict(micro_result.losses)

    for name, value in (
        gradient_metrics.items()
    ):
        if value is not None:
            losses[name] = float(value)

    return VocoderMicroStepResult(
        losses=losses,
        batch_size=micro_result.batch_size,
    )

