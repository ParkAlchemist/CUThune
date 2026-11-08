from pathlib import Path

from shared.src.chiptune_shared.experiment.checkpoint_paths import (
    resolve_checkpoint_selector,
)
from vocoder.scripts.train.resume_config import build_resume_command
from vocoder.scripts.train.run_config import build_run_command


def test_build_vocoder_run_command():
    command = build_run_command(
        python_executable="python",
        train_script=Path("train_vocoder.py"),
        config_path=Path("config.toml"),
    )

    assert command == [
        "python",
        "train_vocoder.py",
        "--config",
        "config.toml",
    ]


def test_build_vocoder_resume_command():
    command = build_resume_command(
        python_executable="python",
        train_script=Path("train_vocoder.py"),
        config_path=Path("config.toml"),
        checkpoint_path=Path("latest.pt"),
    )

    assert command == [
        "python",
        "train_vocoder.py",
        "--config",
        "config.toml",
        "--resume",
        "latest.pt",
    ]


def test_latest_checkpoint_prefers_latest_pt(
    tmp_path: Path,
):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    latest = checkpoint_dir / "latest.pt"
    latest.write_bytes(b"latest")

    numbered = checkpoint_dir / "step_000000100.pt"
    numbered.write_bytes(b"numbered")

    resolved = resolve_checkpoint_selector(
        checkpoint_dir,
        "latest",
        project_root=tmp_path,
    )

    assert resolved == latest.resolve()


def test_latest_checkpoint_falls_back_to_highest_step(
    tmp_path: Path,
):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    low = checkpoint_dir / "step_000000100.pt"
    high = checkpoint_dir / "step_000000500.pt"

    low.write_bytes(b"low")
    high.write_bytes(b"high")

    resolved = resolve_checkpoint_selector(
        checkpoint_dir,
        "latest",
        project_root=tmp_path,
    )

    assert resolved == high.resolve()


def test_resolve_checkpoint_step_selector(
    tmp_path: Path,
):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    expected = checkpoint_dir / "step_000001234.pt"
    expected.write_bytes(b"checkpoint")

    resolved = resolve_checkpoint_selector(
        checkpoint_dir,
        "step:1234",
        project_root=tmp_path,
    )

    assert resolved == expected.resolve()

