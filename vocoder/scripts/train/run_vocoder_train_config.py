from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train" / "train_vocoder.py"

from src.utils.config_launch import (
    resolve_project_path,
    run_command,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch vocoder training from a structured TOML config."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the vocoder experiment TOML file.",
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


def build_run_command(
    *,
    python_executable: str,
    train_script: Path,
    config_path: Path,
    passthrough: list[str] | None = None,
) -> list:
    command = [
        python_executable,
        str(train_script),
        "--config",
        str(config_path),
    ]

    if passthrough:
        command.extend(passthrough)

    return command


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

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(config_path),
    ]

    command.extend(passthrough)

    run_command(
        command=command,
        cwd=PROJECT_ROOT,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

