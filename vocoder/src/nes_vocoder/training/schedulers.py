from __future__ import annotations

import torch

from vocoder.src.nes_vocoder.config.schema import SchedulerConfig
from vocoder.src.nes_vocoder.training.context import TrainingComponents


def scheduler_should_step(
    config: SchedulerConfig,
    *,
    interval: str,
) -> bool:
    return (
        config.enabled
        and config.name != "none"
        and config.interval == interval
    )


def step_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: SchedulerConfig,
    *,
    interval: str,
) -> None:
    if scheduler is None:
        return

    if not scheduler_should_step(
        config,
        interval=interval,
    ):
        return

    scheduler.step()


def step_vocoder_schedulers(
    *,
    components: TrainingComponents,
    generator_config: SchedulerConfig,
    discriminator_config: SchedulerConfig,
    interval: str,
) -> None:
    step_scheduler(
        components.scheduler_generator,
        generator_config,
        interval=interval,
    )

    step_scheduler(
        components.scheduler_discriminator,
        discriminator_config,
        interval=interval,
    )


def get_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
) -> float:
    if not optimizer.param_groups:
        raise RuntimeError(
            "Optimizer contains no parameter groups."
        )

    return float(
        optimizer.param_groups[0]["lr"]
    )


def get_vocoder_learning_rates(
    components: TrainingComponents,
) -> dict[str, float]:
    return {
        "learning_rate_generator": (
            get_optimizer_learning_rate(
                components.optimizers.generator
            )
        ),
        "learning_rate_discriminator": (
            get_optimizer_learning_rate(
                components.optimizers.discriminator
            )
        ),
    }


