from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from vocoder.src.nes_vocoder.config.schema import VocoderExperimentConfig
from src.training.vocoder import (
    TrainingComponents,
    TrainingState,
)


def build_checkpoint_payload(
    *,
    state: TrainingState,
    components: TrainingComponents,
    config: VocoderExperimentConfig,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "epoch": state.epoch,
        "global_step": state.global_step,
        "micro_step": state.micro_step,
        "segments_seen": state.segments_seen,
        "audio_samples_seen": state.audio_samples_seen,
        "generator": (
            components.models.generator.state_dict()
        ),
        "discriminator": (
            components.models.discriminator.state_dict()
        ),
        "optimizer_g": (
            components.optimizers.generator.state_dict()
        ),
        "optimizer_d": (
            components.optimizers.discriminator.state_dict()
        ),
        "scaler": components.scaler.state_dict(),
        "experiment_config": asdict(config),
    }

    if components.scheduler_generator is not None:
        payload["scheduler_g"] = (
            components.scheduler_generator.state_dict()
        )

    if components.scheduler_discriminator is not None:
        payload["scheduler_d"] = (
            components.scheduler_discriminator.state_dict()
        )

    return payload


def save_checkpoint(
    path: Path,
    *,
    state: TrainingState,
    components: TrainingComponents,
    config: VocoderExperimentConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_checkpoint_payload(
        state=state,
        components=components,
        config=config,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def load_scheduler_state(
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    checkpoint_state: dict | None,
    name: str,
) -> None:
    if scheduler is None:
        if checkpoint_state is not None:
            raise ValueError(
                f"Checkpoint contains {name} scheduler state, "
                "but the current configuration disables that "
                "scheduler."
            )

        return

    if checkpoint_state is None:
        raise ValueError(
            f"Current configuration enables the {name} scheduler, "
            "but the checkpoint contains no scheduler state."
        )

    scheduler.load_state_dict(checkpoint_state)


def load_checkpoint(
    path: Path,
    *,
    components: TrainingComponents,
    device: torch.device,
) -> TrainingState:
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    components.models.generator.load_state_dict(
        checkpoint["generator"],
        strict=True,
    )
    components.models.discriminator.load_state_dict(
        checkpoint["discriminator"],
        strict=True,
    )

    components.optimizers.generator.load_state_dict(
        checkpoint["optimizer_g"]
    )
    components.optimizers.discriminator.load_state_dict(
        checkpoint["optimizer_d"]
    )

    load_scheduler_state(
        scheduler=components.scheduler_generator,
        checkpoint_state=checkpoint["scheduler_g"],
        name="generator",
    )

    load_scheduler_state(
        scheduler=components.scheduler_discriminator,
        checkpoint_state=checkpoint["scheduler_d"],
        name="discriminator",
    )

    scaler_state = checkpoint.get("scaler")

    if scaler_state is not None:
        components.scaler.load_state_dict(scaler_state)

    return TrainingState(
        epoch=int(checkpoint.get("epoch", 0)),
        global_step=int(checkpoint.get("global_step", 0)),
        micro_step=int(checkpoint.get("micro_step", 0)),
        segments_seen=int(
            checkpoint.get("segments_seen", 0)
        ),
        audio_samples_seen=int(
            checkpoint.get("audio_samples_seen", 0)
        ),
    )


def prune_numbered_checkpoints(
    checkpoint_dir: Path,
    keep: int,
) -> None:
    paths = sorted(
        checkpoint_dir.glob("step_*.pt"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )

    if keep <= 0:
        for path in paths:
           path.unlink()
           return

    for path in paths[keep:]:
        path.unlink()


