from __future__ import annotations

from pathlib import Path
from typing import Literal, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

from shared.src.chiptune_shared.cqt.cache import CachedCQTTrackDataset
from shared.src.chiptune_shared.cqt.spec import CQTSpec


LengthMode = Literal["max", "min", "source", "target"]


def _mixed_index(
        *,
        seed: int,
        index: int,
        modulus: int,
) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be greater than 0, "
                         f"got {modulus}")

    value = (seed + index * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF

    value ^= value >> 30
    value *= 0xBF58476D1CE4E5B9
    value &= 0xFFFFFFFFFFFFFFFF

    value ^= value >> 27
    value *= 0x94D049BB133111EB
    value &= 0xFFFFFFFFFFFFFFFF

    value ^= value >> 31

    return value % modulus


class UnpairedCQTDataset(Dataset[dict[str, Any]]):
    """
    Unpaired polyphonic-to-2A03 CQT dataset.

    Source domain:
        Arbitrary polyphonic music.

    Target domain:
        Rendered or recorded 2A03-domain music.

    Cross-domain samples are intentionally unpaired.
    """

    def __init__(
        self,
        source_dataset: CachedCQTTrackDataset,
        target_dataset: CachedCQTTrackDataset,
        *,
        cqt_spec: CQTSpec,
        snippet_frames: int = 688,
        length_mode: LengthMode = "max",
        random_target_pairing: bool = True,
        pairing_seed: int = 1337,
    ) -> None:
        if len(source_dataset) == 0:
            raise ValueError("source_dataset must not be empty")
        if len(target_dataset) == 0:
            raise ValueError("target_dataset must not be empty")
        if snippet_frames <= 0:
            raise ValueError("snippet_frames must be greater than 0, "
                             f"got {snippet_frames}")

        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.cqt_spec = cqt_spec
        self.snippet_frames = snippet_frames
        self.length_mode = length_mode
        self.random_target_pairing = random_target_pairing
        self.pairing_seed = pairing_seed

        self.expected_shape = (1, cqt_spec.n_bins, snippet_frames)

    def __len__(self) -> int:
        source_len = len(self.source_dataset)
        target_len = len(self.target_dataset)

        if self.length_mode == "max":
            return max(source_len, target_len)
        elif self.length_mode == "min":
            return min(source_len, target_len)
        elif self.length_mode == "source":
            return source_len
        elif self.length_mode == "target":
            return target_len
        else:
            raise ValueError("Unsupported length mode, "
                             f"got {self.length_mode!r}")

    def _target_index(self, idx: int) -> int:
        target_length = len(self.target_dataset)

        if not self.random_target_pairing:
            return idx % target_length

        return _mixed_index(
            seed=self.pairing_seed,
            index=idx,
            modulus=target_length,
        )

    def _validate_cqt(
            self,
            tensor: torch.Tensor,
            *,
            domain: str,
    ) -> None:
        if tuple(tensor.shape) != self.expected_shape:
            raise ValueError(
                f"{domain} CQT has shape "
                f"{tuple(tensor.shape)} but expected "
                f"shape {self.expected_shape}"
            )

        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"{domain} CQT has nan or inf values."
            )

    @staticmethod
    def _track_id(
            item: dict[str, Any],
            *,
            fallback_index: int,
            domain: str,
    ) -> str:
        if "track_id" in item:
            return str(item["track_id"])

        metadata = item.get("metadata")

        if isinstance(metadata, dict) and metadata.get("track_id") is not None:
            return str(metadata["track_id"])

        if item.get("cache_path") is not None:
            return Path(item["cache_path"]).stem

        return f"{domain}-{fallback_index:08d}"

    def __getitem__(
            self,
            index: int,
    ) -> dict[str, Any]:
        source_index = index % len(self.source_dataset)
        target_index = self._target_index(index)

        source_item = self.source_dataset[source_index]
        target_item = self.target_dataset[target_index]

        source_cqt = source_item["cqt"]
        target_cqt = target_item["cqt"]

        self._validate_cqt(
            source_cqt,
            domain="source",
        )
        self._validate_cqt(
            target_cqt,
            domain="target",
        )

        return {
            "source_cqt": source_cqt,
            "target_cqt": target_cqt,

            "source_energy": float(source_item["energy"]),
            "target_energy": float(target_item["energy"]),

            "source_track_id": self._track_id(
                source_item,
                fallback_index=source_index,
                domain="source",
            ),
            "target_track_id": self._track_id(
                target_item,
                fallback_index=target_index,
                domain="target",
            ),

            "source_frame_start": int(source_item["frame_start"]),
            "target_frame_start": int(target_item["frame_start"]),

            "source_frame_end": int(source_item["frame_end"]),
            "target_frame_end": int(target_item["frame_end"]),

            "source_cache_path": str(source_item["cache_path"]),
            "target_cache_path": str(target_item["cache_path"]),

            "source_path": str(source_item["source_path"]) if source_item.get("source_path") is not None else None,
            "target_path": str(target_item["target_path"]) if target_item.get("target_path") is not None else None,

            "source_valid_mask": torch.ones(self.expected_shape[-1], dtype=torch.bool),
            "target_valid_mask": torch.ones(self.expected_shape[-1], dtype=torch.bool),
        }


def build_unpaired_cqt_dataset(
    cache_root: Path,
    *,
    cqt_spec: CQTSpec,
    snippet_frames: int = 688,
    windows_per_track: int = 16,
    min_window_energy: float = 0.01,
    track_cache_size: int = 8,
    split: Literal["train", "val", "all"] = "train",
    length_mode: LengthMode = "max",
    val_fraction: float = 0.1,
    split_seed: int = 1337,
    pairing_seed: int = 1337,
    random_track: bool = False,
    return_metadata: bool = True,
    return_chroma: bool = False,
    return_phase: bool = False,
) -> UnpairedCQTDataset:
    cache_root = Path(cache_root)

    is_training = split == "train"

    common_arguments = {
        "snippet_frames": snippet_frames,
        "sample_rate": cqt_spec.sample_rate,
        "hop_length": cqt_spec.hop_length,
        "windows_per_track": windows_per_track,
        "random_window": is_training,
        "random_track": random_track,
        "return_chroma": return_chroma,
        "return_phase": return_phase,
        "return_metadata": return_metadata,
        "min_window_energy": min_window_energy,
        "track_cache_size": track_cache_size,
        "split": split,
        "val_fraction": val_fraction,
        "split_seed": split_seed,
    }

    source_dataset = CachedCQTTrackDataset(
        cache_root=cache_root / "poly",
        domain="poly",
        **common_arguments,
    )

    target_dataset = CachedCQTTrackDataset(
        cache_root=cache_root / "chip",
        domain="chip",
        **common_arguments,
    )

    return UnpairedCQTDataset(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        cqt_spec=cqt_spec,
        snippet_frames=snippet_frames,
        length_mode=length_mode,
        random_target_pairing=is_training,
        pairing_seed=pairing_seed,
    )


def build_unpaired_cqt_dataloader(
    cache_root: Path,
    *,
    cqt_spec: CQTSpec,
    batch_size: int = 2,
    num_workers: int = 0,
    snippet_frames: int = 688,
    windows_per_track: int = 16,
    min_window_energy: float = 0.01,
    track_cache_size: int = 8,
    split: Literal["train", "val", "all"] = "train",
    val_fraction: float = 0.1,
    split_seed: int = 1337,
    pairing_seed: int = 1337,
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> DataLoader:
    dataset = build_unpaired_cqt_dataset(
        cache_root=cache_root,
        cqt_spec=cqt_spec,
        snippet_frames=snippet_frames,
        windows_per_track=windows_per_track,
        min_window_energy=min_window_energy,
        track_cache_size=track_cache_size,
        split=split,
        val_fraction=val_fraction,
        split_seed=split_seed,
        pairing_seed=pairing_seed,
    )

    is_training = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            persistent_workers
            and num_workers > 0
        ),
        collate_fn=cqt_collate_fn,
        drop_last=is_training,
    )


_UNCOLLATED_KEYS = {
    "source_cache_path",
    "target_cache_path",
    "source_path",
    "target_path",
    "source_track_id",
    "target_track_id",
}


def cqt_collate_fn(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError(
            "Cannot collate an empty batch."
        )

    expected_keys = set(batch[0])

    for index, item in enumerate(batch[1:], start=1):
        if set(item) != expected_keys:
            missing = expected_keys - set(item)
            extra = set(item) - expected_keys

            raise ValueError(
                f"Inconsistent sample schema at batch "
                f"index {index}: missing={missing}, "
                f"extra={extra}"
            )

    output: dict[str, Any] = {}

    for key in expected_keys:
        values = [item[key] for item in batch]

        if key in _UNCOLLATED_KEYS:
            output[key] = values
            continue

        output[key] = default_collate(values)

    return output
