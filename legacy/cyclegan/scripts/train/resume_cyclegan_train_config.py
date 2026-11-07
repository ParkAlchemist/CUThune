from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_launch import (
    config_to_cli_args,
    format_command,
    load_toml_config,
    resolve_run_checkpoint,
)
import subprocess


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Resume cyclegan training from a TOML config."
    )

    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default="latest")
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

    config = load_toml_config(config_path)

    checkpoint_path = resolve_run_checkpoint(
        project_root=PROJECT_ROOT,
        config=config,
        default_output_root="runs/cyclegan_cqt",
        checkpoint_arg=args.checkpoint,
    )

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train" / "train_cyclegan.py"),
    ]

    command.extend(config_to_cli_args(config))
    command.extend(["--resume", str(checkpoint_path)])
    command.extend(passthrough)

    print("Resuming cyclegan from checkpoint:")
    print(checkpoint_path)
    print("\nGenerated resume command:")
    print(format_command(command))

    if args.dry_run:
        return

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()

