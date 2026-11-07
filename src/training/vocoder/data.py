from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from vocoder.src.nes_vocoder.config.schema import VocoderExperimentConfig
from src.data.vocoder_dataset import CQTVocoderDataset


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


def build_dataloaders(
    config: VocoderExperimentConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    print("Building datasets...")

    data = config.data

    train_dataset = CQTVocoderDataset(
        chip_cache_root=Path(data.chip_cache_root),
        sample_rate=data.sample_rate,
        hop_length=data.hop_length,
        segment_frames=data.segment_frames,
        windows_per_track=data.windows_per_track,
        random_window=data.random_window,
        cache_waveforms=data.cache_waveforms,
    )

    preview_dataset = CQTVocoderDataset(
        chip_cache_root=Path(data.chip_cache_root),
        sample_rate=data.sample_rate,
        hop_length=data.hop_length,
        segment_frames=data.segment_frames,
        windows_per_track=1,
        random_window=False,
        cache_waveforms=data.cache_waveforms,
    )

    pin_memory = (
        data.pin_memory
        and device.type == "cuda"
    )

    train_loader = DataLoader(
        **build_loader_kwargs(
            dataset=train_dataset,
            batch_size=data.batch_size,
            shuffle=True,
            num_workers=data.num_workers,
            drop_last=data.drop_last,
            pin_memory=pin_memory,
            persistent_workers=data.persistent_workers,
            prefetch_factor=data.prefetch_factor,
        )
    )

    preview_loader = DataLoader(
        preview_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=pin_memory,
    )

    return train_loader, preview_loader


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


