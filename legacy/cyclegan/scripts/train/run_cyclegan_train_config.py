from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_launch import (
    build_training_command,
    run_command,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch cyclegan training from a TOML config."
    )

    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")

    args, passthrough = parser.parse_known_args()

    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    return args, passthrough


def main() -> None:
    args, passthrough = parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    command = build_training_command(
        project_root=PROJECT_ROOT,
        train_script=PROJECT_ROOT / "scripts" / "train" / "train_cyclegan.py",
        config_path=config_path,
        passthrough_args=passthrough,
    )

    run_command(
        command=command,
        cwd=PROJECT_ROOT,
        dry_run=args.dry_run,
        label="Generated cyclegan training command",
    )


if __name__ == "__main__":
    main()

