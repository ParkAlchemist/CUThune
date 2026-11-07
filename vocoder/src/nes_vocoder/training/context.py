from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vocoder.src.nes_vocoder.config.schema import VocoderExperimentConfig
from vocoder.src.nes_vocoder.models.src.losses import VocoderLossBundle
from vocoder.src.nes_vocoder.training.metrics import ScalarMetricAccumulator
from vocoder.src.nes_vocoder.training.step import (
    VocoderModels,
    VocoderOptimizers,
)


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    checkpoint_dir: Path
    preview_dir: Path
    tensorboard_dir: Path
    debug_dir: Path
    status_path: Path
    resolved_config_path: Path
    source_config_path: Path


@dataclass
class TrainingComponents:
    models: VocoderModels
    optimizers: VocoderOptimizers
    scheduler_generator: Any
    scheduler_discriminator: Any
    loss_bundle: VocoderLossBundle
    scaler: torch.amp.GradScaler


@dataclass
class TrainingState:
    epoch: int = 0
    global_step: int = 0
    micro_step: int = 0
    segments_seen: int = 0
    audio_samples_seen: int = 0

    latest_checkpoint: str | None = None
    latest_preview: str | None = None

    accumulation_in_progress: bool = False


@dataclass
class TrainingRuntime:
    config: VocoderExperimentConfig
    config_path: Path
    device: torch.device
    paths: RunPaths
    components: TrainingComponents
    train_loader: Any
    preview_loader: Any
    overfit_batches: list[dict] | None
    writer: Any
    stop_controller: Any

    metric_accumulator: ScalarMetricAccumulator
    resumed_from_checkpoint: bool = False

