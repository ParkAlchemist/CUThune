from __future__ import annotations

import argparse
from pathlib import Path
import sys

from vocoder.src.nes_vocoder.training.checkpointing import load_checkpoint
from vocoder.src.nes_vocoder.training.runtime import resolve_project_path, validate_implemented_features, set_seed, \
    resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from vocoder.src.nes_vocoder.config.loader import (
    load_vocoder_config,
)
from vocoder.src.nes_vocoder.config.validator import (
    validate_vocoder_config,
)
from vocoder.src.nes_vocoder.training.engine import run_training
from vocoder.src.nes_vocoder.training.logging import (
    save_config_snapshot,
)
from vocoder.src.nes_vocoder.training.setup import (
    build_training_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CQT-conditioned components."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = resolve_project_path(
        args.config,
        PROJECT_ROOT,
    )

    config = load_vocoder_config(config_path)
    validate_vocoder_config(config, check_paths=True)
    validate_implemented_features(config)

    set_seed(config.run.seed)
    device = resolve_device(config.run.device)

    runtime = build_training_runtime(
        config=config,
        config_path=config_path,
        device=device,
    )

    save_config_snapshot(
        config=config,
        source_path=config_path,
        paths=runtime.paths,
    )

    state = None

    if args.resume is not None:
        resume_path = resolve_project_path(
            args.resume,
            PROJECT_ROOT,
        )

        print(f"Resuming from: {resume_path}")

        state = load_checkpoint(
            resume_path,
            components=runtime.components,
            device=device,
        )

        print(
            "Resumed at "
            f"epoch={state.epoch}, "
            f"global_step={state.global_step}"
        )

        runtime.resumed_from_checkpoint = True

    if state is None:
        from vocoder.src.nes_vocoder.training.context import TrainingState

        state = TrainingState()

    try:
        run_training(runtime, state)
    finally:
        if runtime.writer is not None:
            runtime.writer.close()

    print("Vocoder training complete.")


if __name__ == "__main__":
    main()

