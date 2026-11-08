from __future__ import annotations

import math

import torch

from vocoder.src.nes_vocoder.config.schema import (
    OptimizerConfig,
    SchedulerConfig,
    VocoderExperimentConfig,
)
from vocoder.src.nes_vocoder.config.adapter import build_loss_config
from vocoder.src.nes_vocoder.models.discriminators import (
    HiFiGANMultiDiscriminator,
)
from vocoder.src.nes_vocoder.models.generator import (
    CQTUHiFiGANGenerator,
)
from vocoder.src.nes_vocoder.training.step import (
    VocoderModels,
    VocoderOptimizers,
)
from vocoder.src.nes_vocoder.training.context import TrainingComponents
from vocoder.src.nes_vocoder.losses.objective import VocoderLossBundle


def build_optimizer(
    parameters,
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    if config.name == "adam":
        optimizer_class = torch.optim.Adam
    elif config.name == "adamw":
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(
            f"Unsupported optimizer: {config.name!r}"
        )

    return optimizer_class(
        parameters,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        eps=config.eps,
    )


def build_fixed_floor_cosine_scheduler(
        optimizer: torch.optim.Optimizer,
        *,
        t_max: int,
        eta_min: float,
) -> torch.optim.lr_scheduler.LambdaLR:

    if t_max <= 0:
        raise ValueError(
            f"t_max must be greater than 0. Got {t_max}."
        )

    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    if len(set(base_lrs)) != 1:
        raise ValueError(
            f"base_lrs must contain only one element. Got {base_lrs}"
        )

    base_lr = base_lrs[0]

    if not 0.0 <= eta_min <= base_lr:
        raise ValueError(
           f"eta_min must be non-negative and smaller than base_lr={base_lr}, got eta_min={eta_min}"
        )

    minimum_multiplier = eta_min / base_lr

    def lr_multiplier(step: int) -> float:
        progress = min(
            max(step, 0),
            t_max,
        ) / t_max

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return minimum_multiplier + (1.0 - minimum_multiplier) * cosine

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_multiplier,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
):
    if not config.enabled or config.name == "none":
        return None

    if config.name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.gamma,
        )

    if config.name == "cosine":
        return build_fixed_floor_cosine_scheduler(
            optimizer,
            t_max=config.t_max,
            eta_min=config.eta_min,
        )

    raise ValueError(
        f"Unsupported scheduler: {config.name!r}"
    )


def build_training_components(
    config: VocoderExperimentConfig,
    device: torch.device,
) -> TrainingComponents:
    print("Building models...")

    generator = CQTUHiFiGANGenerator(
        config.data.cqt_bins,
        config.generator,
    ).to(device)

    generator.to(device)

    discriminator = HiFiGANMultiDiscriminator(
        config.discriminator,
    ).to(device)

    models = VocoderModels(
        generator=generator,
        discriminator=discriminator,
    )

    optimizers = VocoderOptimizers(
        generator=build_optimizer(
            generator.parameters(),
            config.optimizer_generator,
        ),
        discriminator=build_optimizer(
            discriminator.parameters(),
            config.optimizer_discriminator,
        ),
    )

    scheduler_generator = build_scheduler(
        optimizers.generator,
        config.scheduler_generator,
    )

    scheduler_discriminator = build_scheduler(
        optimizers.discriminator,
        config.scheduler_discriminator,
    )

    loss_bundle = VocoderLossBundle(
        config=build_loss_config(config),
        sample_rate=config.data.sample_rate,
    ).to(device)

    scaler_enabled = (
        config.amp.enabled
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=scaler_enabled,
        init_scale=config.amp.initial_scale,
        growth_interval=config.amp.growth_interval,
    )

    return TrainingComponents(
        models=models,
        optimizers=optimizers,
        scheduler_generator=scheduler_generator,
        scheduler_discriminator=scheduler_discriminator,
        loss_bundle=loss_bundle,
        scaler=scaler,
    )

