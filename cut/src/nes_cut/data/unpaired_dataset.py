from __future__ import annotations

import random
from pathlib import Path
from typing import Literal, Any

from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

from shared.src.chiptune_shared.cqt.cache import CachedCQTTrackDataset


class UnpairedCQTDataset(Dataset):
    """
    CycleGAN-ready unpaired dataset.

    X = poly
    Y = chip

    Returned keys:
        real_x / real_poly
        real_y / real_chip

    Also returns optional chroma/phase with matching domain prefixes if available.
    """

    def __init__(
        self,
        poly_dataset: CachedCQTTrackDataset,
        chip_dataset: CachedCQTTrackDataset,
        length_mode: Literal["max", "min", "poly", "chip"] = "max",
        random_y: bool = True,
    ) -> None:
        self.poly_dataset = poly_dataset
        self.chip_dataset = chip_dataset
        self.length_mode = length_mode
        self.random_y = random_y

    def __len__(self) -> int:
        if self.length_mode == "max":
            return max(len(self.poly_dataset), len(self.chip_dataset))
        if self.length_mode == "min":
            return min(len(self.poly_dataset), len(self.chip_dataset))
        if self.length_mode == "poly":
            return len(self.poly_dataset)
        if self.length_mode == "chip":
            return len(self.chip_dataset)

        raise ValueError(f"Unknown length_mode: {self.length_mode}")

    def _chip_index(self, idx: int) -> int:
        if self.random_y:
            return random.randint(0, len(self.chip_dataset) - 1)
        return idx % len(self.chip_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        poly_item = self.poly_dataset[idx % len(self.poly_dataset)]
        chip_item = self.chip_dataset[self._chip_index(idx)]

        batch_item: dict[str, Any] = {
            "real_x": poly_item["cqt"],
            "real_y": chip_item["cqt"],
            "real_poly": poly_item["cqt"],
            "real_chip": chip_item["cqt"],

            "energy_poly": float(poly_item["energy"]),
            "energy_chip": float(chip_item["energy"]),

            "frame_start_poly": poly_item["frame_start"],
            "frame_end_poly": poly_item["frame_end"],
            "frame_start_chip": chip_item["frame_start"],
            "frame_end_chip": chip_item["frame_end"],

            "cache_path_poly": poly_item["cache_path"],
            "cache_path_chip": chip_item["cache_path"],
        }

        if "chroma" in poly_item:
            batch_item["chroma_x"] = poly_item["chroma"]
            batch_item["chroma_poly"] = poly_item["chroma"]

        if "chroma" in chip_item:
            batch_item["chroma_y"] = chip_item["chroma"]
            batch_item["chroma_chip"] = chip_item["chroma"]

        if "phase" in poly_item:
            batch_item["phase_x"] = poly_item["phase"]
            batch_item["phase_poly"] = poly_item["phase"]

        if "phase" in chip_item:
            batch_item["phase_y"] = chip_item["phase"]
            batch_item["phase_chip"] = chip_item["phase"]

        if "metadata" in poly_item:
            batch_item["metadata_poly"] = poly_item["metadata"]

        if "metadata" in chip_item:
            batch_item["metadata_chip"] = chip_item["metadata"]

        if "source_path" in poly_item:
            batch_item["source_path_poly"] = poly_item["source_path"]

        if "source_path" in chip_item:
            batch_item["source_path_chip"] = chip_item["source_path"]

        return batch_item


def build_unpaired_cqt_dataset(
    cache_root: Path,
    snippet_seconds: float = 4.0,
    snippet_frames: int | None = None,
    sample_rate: int = 22050,
    hop_length: int = 512,
    windows_per_track: int = 16,
    return_chroma: bool = True,
    return_phase: bool = False,
    return_metadata: bool = False,
    min_window_energy: float = 0.01,
    track_cache_size: int = 8,
    split: Literal["train", "val", "all"] = "train",
    val_fraction: float = 0.1,
    split_seed: int = 1337,
) -> UnpairedCQTDataset:
    cache_root = Path(cache_root)

    poly_dataset = CachedCQTTrackDataset(
        cache_root=cache_root / "poly",
        domain="poly",
        snippet_seconds=snippet_seconds,
        snippet_frames=snippet_frames,
        sample_rate=sample_rate,
        hop_length=hop_length,
        windows_per_track=windows_per_track,
        random_window=True,
        random_track=False,
        return_chroma=return_chroma,
        return_phase=return_phase,
        return_metadata=return_metadata,
        min_window_energy=min_window_energy,
        track_cache_size=track_cache_size,
        split=split,
        val_fraction=val_fraction,
        split_seed=split_seed,
    )

    chip_dataset = CachedCQTTrackDataset(
        cache_root=cache_root / "chip",
        domain="chip",
        snippet_seconds=snippet_seconds,
        snippet_frames=snippet_frames,
        sample_rate=sample_rate,
        hop_length=hop_length,
        windows_per_track=windows_per_track,
        random_window=True,
        random_track=False,
        return_chroma=return_chroma,
        return_phase=return_phase,
        return_metadata=return_metadata,
        min_window_energy=min_window_energy,
        track_cache_size=track_cache_size,
        split=split,
        val_fraction=val_fraction,
        split_seed=split_seed,
    )

    return UnpairedCQTDataset(
        poly_dataset=poly_dataset,
        chip_dataset=chip_dataset,
        length_mode="max",
        random_y=True,
    )


def build_unpaired_cqt_dataloader(
    cache_root: Path,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True,
    snippet_seconds: float = 4.0,
    snippet_frames: int | None = None,
    sample_rate: int = 22050,
    hop_length: int = 512,
    windows_per_track: int = 16,
    return_chroma: bool = True,
    return_phase: bool = False,
    return_metadata: bool = False,
    min_window_energy: float = 0.01,
    track_cache_size: int = 8,
    split: Literal["train", "val", "all"] = "train",
    val_fraction: float = 0.1,
    split_seed: int = 1337,
) -> DataLoader:
    dataset = build_unpaired_cqt_dataset(
        cache_root=cache_root,
        snippet_seconds=snippet_seconds,
        snippet_frames=snippet_frames,
        sample_rate=sample_rate,
        hop_length=hop_length,
        windows_per_track=windows_per_track,
        return_chroma=return_chroma,
        return_phase=return_phase,
        return_metadata=return_metadata,
        min_window_energy=min_window_energy,
        track_cache_size=track_cache_size,
        split=split,
        val_fraction=val_fraction,
        split_seed=split_seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=cqt_collate_fn,
        drop_last=True,
    )


def cqt_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Custom collate for CQT datasets.

    - Tensors are stacked normally.
    - Numbers are collated normally.
    - Strings/paths are kept as lists.
    - Metadata dicts are kept as lists to avoid PyTorch default_collate
      crashing on nested None values such as fmin=None.
    - None values are kept as lists or ignored depending on key usage.
    """
    if not batch:
        return {}

    output: dict[str, Any] = {}

    keys = set()
    for item in batch:
        keys.update(item.keys())

    for key in keys:
        values = [item.get(key) for item in batch]

        # Metadata can contain None, nested dicts, paths, etc.
        # Keep it uncollated.
        if key.startswith("metadata"):
            output[key] = values
            continue

        # Source/cache paths should remain readable strings.
        if key.startswith("source_path") or key.startswith("cache_path"):
            output[key] = values
            continue

        # If any value is None, default_collate will crash.
        # Keep as list.
        if any(v is None for v in values):
            output[key] = values
            continue

        try:
            output[key] = default_collate(values)
        except TypeError:
            output[key] = values

    return output
