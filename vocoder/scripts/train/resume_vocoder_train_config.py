from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys

import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train" / "train_vocoder.py"

sys.path.insert(0, str(PROJECT_ROOT))

from src.training.checkpoint_paths import (
    extract_numbered_checkpoint_value,
    find_latest_numbered_checkpoint,
    resolve_checkpoint_selector,
)

from src.utils.config_launch import resolve_project_path, run_command

from src.config.vocoder_config_loader import load_vocoder_config


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Resume vocoder training from a checkpoint."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the vocoder experiment TOML file.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help=(
            "Checkpoint selector. Supported values: "
            "'latest', 'step:N', 'epoch:N', or an explicit path."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated command without executing it.",
    )

    args, passthrough = parser.parse_known_args()

    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    return args, passthrough


def build_resume_command(
    *,
    python_executable: str,
    train_script: Path,
    config_path: Path,
    checkpoint_path: Path,
    passthrough: list[str] | None = None,
) -> list:
    command = [
        python_executable,
        str(train_script),
        "--config",
        str(config_path),
        "--resume",
        str(checkpoint_path),
    ]

    if passthrough:
        command.extend(passthrough)

    return command


def inspect_checkpoint(path: Path) -> tuple[int | None, int | None]:
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not read checkpoint: {path}"
        ) from exc

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Expected checkpoint dictionary, got {type(checkpoint).__name__}."
        )

    epoch = checkpoint.get("epoch")
    global_step = checkpoint.get("global_step")

    return epoch, global_step


def format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def main() -> None:
    args, passthrough = parse_args()

    config_path = resolve_project_path(args.config, PROJECT_ROOT)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Vocoder configuration file not found: {config_path}"
        )

    if not config_path.is_file():
        raise ValueError(
            f"Vocoder configuration path is not a file: {config_path}"
        )

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(
            f"Vocoder training script not found: {TRAIN_SCRIPT}"
        )

    config = load_vocoder_config(config_path)

    output_root = Path(config.run.output_root).expanduser()

    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    run_dir = (
        output_root.resolve()
        / config.run.experiment_name
    )

    checkpoint_dir = run_dir / "checkpoints"

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}"
        )

    checkpoint_path = resolve_checkpoint_selector(
        checkpoint_dir=checkpoint_dir,
        selector=args.checkpoint,
        project_root=PROJECT_ROOT,
    )

    epoch, global_step = inspect_checkpoint(checkpoint_path)

    print("Resuming vocoder from checkpoint:")
    print(checkpoint_path)

    if epoch is not None or global_step is not None:
        print(
            "Checkpoint state: "
            f"epoch={epoch!r}, global_step={global_step!r}"
        )

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(config_path),
        "--resume",
        str(checkpoint_path),
    ]

    command.extend(passthrough)

    print("\nGenerated resume command:")
    print(format_command(command))

    if args.dry_run:
        return

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()

