from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")


def accumulation_windows(
    iterable: Iterable[T],
    window_size: int,
) -> Iterator[list[T]]:
    if window_size < 1:
        raise ValueError(
            f"window_size must be at least 1, got {window_size}."
        )

    window: list[T] = []

    for item in iterable:
        window.append(item)

        if len(window) == window_size:
            yield window
            window = []

    if window:
        yield window


def average_loss_dicts(
    loss_dicts: list[dict[str, float]],
) -> dict[str, float]:
    if not loss_dicts:
        raise ValueError(
            "Cannot average an empty collection of losses."
        )

    expected_keys = set(loss_dicts[0])

    for index, losses in enumerate(loss_dicts[1:], start=1):
        if set(losses) != expected_keys:
            raise ValueError(
                "Micro-step loss dictionaries contain different "
                f"keys at index {index}."
            )

    return {
        key: sum(losses[key] for losses in loss_dicts)
        / len(loss_dicts)
        for key in expected_keys
    }

