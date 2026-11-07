from __future__ import annotations

from pathlib import Path
import random
import signal

import numpy as np
import torch

from vocoder.src.nes_vocoder.config.schema import (
    ControlConfig,
    VocoderExperimentConfig,
)
from src.training.vocoder.context import RunPaths


class StopController:
    def __init__(self) -> None:
        self.stop_requested = False

    def request_stop(
        self,
        signum=None,
        frame=None,
    ) -> None:
        print(
            "\n[CONTROL] Stop requested. "
            "Saving checkpoint before exit..."
        )
        self.stop_requested = True


def derive_training_seed(
        *,
        base_seed: int,
        epoch: int,
        resume_global_step: int = 0,
) -> int:
    if base_seed < 0:
        raise ValueError(f"base_seed must be non-negative, got {base_seed}")

    if epoch < 0:
        raise ValueError(f"epoch must be non-negative, got {epoch}")

    if resume_global_step < 0:
        raise ValueError(f"resume_global_step must be non-negative, got {resume_global_step}")

    return (base_seed + epoch + resume_global_step) % (2**32)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_project_path(
    path: Path,
    project_root: Path,
) -> Path:
    path = path.expanduser()

    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")

    device = torch.device(requested)

    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(device)
        required_arch = f"sm_{capability[0]}{capability[1]}"
        available_arches = torch.cuda.get_arch_list()

        if required_arch not in available_arches:
            raise RuntimeError(
                "The installed PyTorch build does not support the "
                "selected GPU.\n"
                f"GPU: {torch.cuda.get_device_name(device)}\n"
                f"Required architecture: {required_arch}\n"
                f"Available architectures: {available_arches}"
            )

        # Execute an actual kernel, not only is_available().
        test_tensor = torch.ones(1, device=device)
        test_result = test_tensor + 1
        torch.cuda.synchronize(device)

        if test_result.item() != 2.0:
            raise RuntimeError("CUDA pre-flight kernel failed.")

    return device


def build_run_paths(
    config: VocoderExperimentConfig,
) -> RunPaths:
    run_dir = (
        Path(config.run.output_root)
        / config.run.experiment_name
    )

    return RunPaths(
        run_dir=run_dir,
        checkpoint_dir=run_dir / "checkpoints",
        preview_dir=run_dir / "previews",
        tensorboard_dir=run_dir / "tensorboard",
        debug_dir=run_dir / "debug",
        status_path=run_dir / "status.json",
        resolved_config_path=run_dir / "config_resolved.json",
        source_config_path=run_dir / "config_source.toml",
    )


def create_run_directories(paths: RunPaths) -> None:
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths.preview_dir.mkdir(parents=True, exist_ok=True)


def configure_stop_controller(
    config: ControlConfig,
) -> StopController:
    controller = StopController()

    if config.enabled:
        signal.signal(
            signal.SIGINT,
            controller.request_stop,
        )

    return controller


def validate_implemented_features(
    config: VocoderExperimentConfig,
) -> None:
    training = config.training

    unsupported = []

    if training.generator_start_step != 0:
        unsupported.append(
            "training.generator_start_step"
        )

    if training.discriminator_start_step != 0:
        unsupported.append(
            "training.discriminator_start_step"
        )

    if training.adversarial_start_step != 0:
        unsupported.append(
            "training.adversarial_start_step"
        )

    if config.augmentation.enabled:
        unsupported.append("augmentation.enabled")

    if unsupported:
        raise NotImplementedError(
            "The following configured features are not yet "
            "implemented\n"
            + "\n".join(
                f"  - {field}"
                for field in unsupported
            )
        )

