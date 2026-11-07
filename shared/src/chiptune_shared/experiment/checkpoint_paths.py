from __future__ import annotations

from pathlib import Path


def extract_numbered_checkpoint_value(
    path: Path,
    prefix: str,
) -> int:
    stem_prefix = f"{prefix}_"

    if not path.stem.startswith(stem_prefix):
        return -1

    value = path.stem[len(stem_prefix):]

    try:
        return int(value)
    except ValueError:
        return -1


def find_latest_numbered_checkpoint(
    checkpoint_dir: Path,
    *,
    prefix: str,
) -> Path | None:
    paths = checkpoint_dir.glob(f"{prefix}_*.pt")

    valid_paths = [
        path
        for path in paths
        if extract_numbered_checkpoint_value(
            path,
            prefix,
        ) >= 0
    ]

    if not valid_paths:
        return None

    return max(
        valid_paths,
        key=lambda path: extract_numbered_checkpoint_value(
            path,
            prefix,
        ),
    )


def resolve_checkpoint_selector(
    checkpoint_dir: Path,
    selector: str,
    *,
    project_root: Path,
) -> Path:
    if selector == "latest":
        latest_path = checkpoint_dir / "latest.pt"

        if latest_path.is_file():
            return latest_path.resolve()

        for prefix in ("step", "epoch", "stop_step"):
            path = find_latest_numbered_checkpoint(
                checkpoint_dir,
                prefix=prefix,
            )

            if path is not None:
                return path.resolve()

        raise FileNotFoundError(
            f"No checkpoint found under: {checkpoint_dir}"
        )

    if selector.startswith("step:"):
        step = int(selector.split(":", 1)[1])
        path = checkpoint_dir / f"step_{step:09d}.pt"

    elif selector.startswith("epoch:"):
        epoch = int(selector.split(":", 1)[1])
        path = checkpoint_dir / f"epoch_{epoch:04d}.pt"

    else:
        path = Path(selector).expanduser()

        if not path.is_absolute():
            path = project_root / path

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    return path

