from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from vocoder.src.nes_vocoder.config.schema import VocoderExperimentConfig, VocoderWindowSamplingConfig
from vocoder.src.nes_vocoder.data.dataset import build_vocoder_datasets


@dataclass(frozen=True)
class VocoderDataLoaders:
    train: DataLoader
    validation: DataLoader


def build_loader_kwargs(
    *,
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor

    return kwargs


def build_vocoder_dataloaders(
    config: VocoderExperimentConfig,
    *,
    split_manifest_path: Path,
) -> VocoderDataLoaders:
    sampling = (
        VocoderWindowSamplingConfig(
            random_probability=(
                config.data.sampling
                .random_probability
            ),
            active_to_silent_probability=(
                config.data.sampling
                .active_to_silent_probability
            ),
            silent_to_active_probability=(
                config.data.sampling
                .silent_to_active_probability
            ),
            silence_rich_probability=(
                config.data.sampling
                .silence_rich_probability
            ),
            silence_rms_threshold=(
                config.data.sampling
                .silence_rms_threshold
            ),
            silence_rich_fraction=(
                config.data.sampling
                .silence_rich_fraction
            ),
            boundary_jitter_frames=(
                config.data.sampling
                .boundary_jitter_frames
            ),
            minimum_state_frames=(
                config.data.sampling
                .minimum_state_frames
            ),
        )
    )

    (
        train_dataset,
        validation_dataset,
    ) = build_vocoder_datasets(
        chip_cache_root=Path(
            config.data.chip_cache_root
        ),
        sample_rate=(
            config.data.sample_rate
        ),
        hop_length=(
            config.data.hop_length
        ),
        segment_frames=(
            config.data.segment_frames
        ),
        train_windows_per_track=(
            config.data
            .train_windows_per_track
        ),
        validation_windows_per_track=(
            config.data
            .validation_windows_per_track
        ),
        validation_fraction=(
            config.data.validation_fraction
        ),
        split_seed=(
            config.data.split_seed
        ),
        cache_waveforms=(
            config.data.cache_waveforms
        ),
        sampling=sampling,
        split_manifest_path=(
            split_manifest_path
        ),
    )

    loader_arguments = {
        "num_workers": (
            config.data.num_workers
        ),
        "pin_memory": (
            config.data.pin_memory
        ),
        "persistent_workers": (
            config.data.persistent_workers
            and config.data.num_workers > 0
        ),
    }

    if config.data.num_workers > 0:
        loader_arguments[
            "prefetch_factor"
        ] = config.data.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=(
            config.data.batch_size
        ),
        shuffle=True,
        drop_last=True,
        **loader_arguments,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=(
            config.data
            .validation_batch_size
        ),
        shuffle=False,
        drop_last=False,
        **loader_arguments,
    )

    return VocoderDataLoaders(
        train=train_loader,
        validation=validation_loader,
    )


def collect_overfit_batches(
    train_loader: DataLoader,
    count: int,
) -> list[dict] | None:
    if count <= 0:
        return None

    print(
        f"Overfit mode enabled: using first {count} "
        "batches repeatedly."
    )

    iterator = iter(train_loader)
    batches: list[dict] = []

    for _ in range(count):
        try:
            batches.append(next(iterator))
        except StopIteration as exc:
            raise RuntimeError(
                "Training loader contains fewer batches than "
                f"overfit_batches={count}."
            ) from exc

    return batches


def save_overfit_debug_data(
    batches: list[dict] | None,
    debug_dir: Path,
) -> None:
    if not batches:
        return

    debug_dir.mkdir(parents=True, exist_ok=True)

    batch_path = debug_dir / "overfit_batches.pt"
    metadata_path = debug_dir / "overfit_batches_metadata.json"

    torch.save(batches, batch_path)

    metadata_rows: list[dict[str, Any]] = []

    for batch_index, batch in enumerate(batches):
        batch_size = int(batch["cqt"].shape[0])

        source_paths = batch.get(
            "source_path",
            [""] * batch_size,
        )

        for item_index in range(batch_size):
            metadata_rows.append(
                {
                    "batch_index": batch_index,
                    "item_index": item_index,
                    "source_path": source_paths[item_index],
                    "frame_start": int(
                        batch["frame_start"][item_index]
                    ),
                    "frame_end": int(
                        batch["frame_end"][item_index]
                    ),
                    "sample_start": int(
                        batch["sample_start"][item_index]
                    ),
                    "sample_end": int(
                        batch["sample_end"][item_index]
                    ),
                }
            )

    metadata_path.write_text(
        json.dumps(
            metadata_rows,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Saved fixed overfit batches: {batch_path}")
    print(f"Saved fixed overfit metadata: {metadata_path}")


