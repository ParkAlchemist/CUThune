from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset

from shared.src.chiptune_shared.cqt.spec import seconds_to_frames, pad_feature_last_dim, cqt_window_energy


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def read_csv_index(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def discover_cache_records(cache_root: Path) -> list[dict[str, Any]]:
    cache_root = Path(cache_root)

    index_jsonl = cache_root / "index.jsonl"
    index_csv = cache_root / "index.csv"

    if index_jsonl.exists():
        records = read_jsonl(index_jsonl)
    elif index_csv.exists():
        records = read_csv_index(index_csv)
    else:
        records = [
            {
                "cache_path": str(path),
                "status": "ok",
            }
            for path in sorted(cache_root.glob("*.pt"))
        ]

    usable: list[dict[str, Any]] = []

    for record in records:
        if record.get("status") not in (None, "ok", "skipped_existing"):
            continue

        cache_path = record.get("cache_path")
        if cache_path is None:
            continue

        cache_path = Path(cache_path)
        if not cache_path.exists():
            continue

        record = dict(record)
        record["cache_path"] = str(cache_path)
        usable.append(record)

    if not usable:
        raise RuntimeError(f"No usable cache records found under: {cache_root}")

    return usable


def stable_record_split_value(record: dict[str, Any], split_seed: int = 1337) -> float:
    """
    Deterministically maps a cache/source record to [0, 1).

    Used for stable train/validation splitting at track level.
    """
    key = (
        record.get("source_path")
        or record.get("cache_path")
        or record.get("id")
        or json.dumps(record, sort_keys=True)
    )

    text = f"{split_seed}:{key}"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()

    # Use first 12 hex chars for a stable integer.
    value = int(digest[:12], 16)
    max_value = float(16 ** 12)

    return value / max_value


def split_cache_records(
    records: list[dict[str, Any]],
    split: Literal["train", "val", "all"] = "train",
    val_fraction: float = 0.1,
    split_seed: int = 1337,
) -> list[dict[str, Any]]:
    """
    Deterministic track-level split.

    split="train":
        keeps records with split_value >= val_fraction

    split="val":
        keeps records with split_value < val_fraction

    split="all":
        keeps all records
    """
    if split == "all":
        return records

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    selected: list[dict[str, Any]] = []

    for record in records:
        value = stable_record_split_value(record, split_seed=split_seed)

        if split == "val" and value < val_fraction:
            selected.append(record)
        elif split == "train" and value >= val_fraction:
            selected.append(record)

    if not selected:
        raise RuntimeError(
            f"No records selected for split={split}. "
            f"records={len(records)}, val_fraction={val_fraction}"
        )

    return selected


class SmallTensorLRUCache:
    """
    Tiny in-memory LRU cache for loaded .pt track payloads.

    This is useful because random snippet sampling can repeatedly hit the same
    track within an epoch. Keeping a few recent tracks avoids excessive disk IO.
    """

    def __init__(self, max_items: int = 8) -> None:
        self.max_items = max(0, int(max_items))
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, path: Path) -> dict[str, Any]:
        key = str(path)

        if self.max_items <= 0:
            return torch.load(path, map_location="cpu")

        if key in self._items:
            value = self._items.pop(key)
            self._items[key] = value
            return value

        value = torch.load(path, map_location="cpu")

        self._items[key] = value

        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

        return value


class CachedCQTTrackDataset(Dataset):
    """
    Dataset over full-track cached CQT files.

    Each __getitem__ returns a fixed-length snippet sampled from a cached full-track CQT.

    Expected cache payload keys:
        cqt_mag: Tensor [1, n_bins, T]
        chroma: optional Tensor [12, T]
        cqt_phase: optional Tensor [n_bins, T]
        metadata: dict

    Returned item:
        {
            "cqt": Tensor [1, n_bins, snippet_frames],
            "chroma": optional Tensor [12, snippet_frames],
            "phase": optional Tensor [n_bins, snippet_frames],
            "metadata": dict,
            "cache_path": str,
            "source_path": optional str,
            "frame_start": int,
            "frame_end": int,
            "track_index": int,
            "domain": str
        }
    """

    def __init__(
            self,
            cache_root: Path,
            domain: Domain,
            snippet_seconds: float = 4.0,
            snippet_frames: int | None = None,
            sample_rate: int = 22050,
            hop_length: int = 512,
            windows_per_track: int = 16,
            random_window: bool = True,
            random_track: bool = False,
            return_chroma: bool = True,
            return_phase: bool = False,
            return_metadata: bool = True,
            min_window_energy: float = 0.01,
            max_resample_attempts: int = 10,
            pad_short_tracks: bool = True,
            track_cache_size: int = 8,
            dtype: torch.dtype = torch.float32,
            expected_config_hash: str | None = None,
            require_config_hash_match: bool = False,
            split: Literal["train", "val", "all"] = "train",
            val_fraction: float = 0.1,
            split_seed: int = 1337,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.domain = domain
        all_records = discover_cache_records(self.cache_root)
        self.records = split_cache_records(
            all_records,
            split=split,
            val_fraction=val_fraction,
            split_seed=split_seed,
        )

        self.sample_rate = sample_rate
        self.hop_length = hop_length

        self.snippet_frames = (
            snippet_frames
            if snippet_frames is not None
            else seconds_to_frames(snippet_seconds, sample_rate, hop_length)
        )

        self.windows_per_track = max(1, int(windows_per_track))
        self.random_window = random_window
        self.random_track = random_track

        self.return_chroma = return_chroma
        self.return_phase = return_phase
        self.return_metadata = return_metadata

        self.min_window_energy = float(min_window_energy)
        self.max_resample_attempts = max(1, int(max_resample_attempts))
        self.pad_short_tracks = pad_short_tracks

        self.dtype = dtype
        self.expected_config_hash = expected_config_hash
        self.require_config_hash_match = require_config_hash_match

        self.track_cache = SmallTensorLRUCache(max_items=track_cache_size)

        if self.require_config_hash_match and self.expected_config_hash is None:
            raise ValueError(
                "require_config_hash_match=True requires expected_config_hash."
            )

    def __len__(self) -> int:
        return len(self.records) * self.windows_per_track

    def _record_index_for_item(self, idx: int) -> int:
        if self.random_track:
            return random.randint(0, len(self.records) - 1)

        return idx % len(self.records)

    def _load_payload(self, record_index: int) -> tuple[dict[str, Any], Path]:
        record = self.records[record_index]
        cache_path = Path(record["cache_path"])
        payload = self.track_cache.get(cache_path)

        if "cqt_mag" not in payload:
            raise KeyError(f"Cache file missing 'cqt_mag': {cache_path}")

        if self.expected_config_hash is not None:
            found_hash = payload.get("config_hash") or payload.get("metadata", {}).get("config_hash")
            if found_hash != self.expected_config_hash and self.require_config_hash_match:
                raise RuntimeError(
                    f"Config hash mismatch in {cache_path}. "
                    f"Expected {self.expected_config_hash}, found {found_hash}."
                )

        return payload, cache_path

    def _choose_start_frame(self, total_frames: int, idx: int) -> int:
        if total_frames <= self.snippet_frames:
            return 0

        max_start = total_frames - self.snippet_frames

        if self.random_window:
            return random.randint(0, max_start)

        # Deterministic spread across the track.
        virtual_window_idx = idx // max(1, len(self.records))
        if self.windows_per_track <= 1:
            return 0

        fraction = virtual_window_idx / max(1, self.windows_per_track - 1)
        return int(round(max_start * fraction))

    def _slice_last_dim(
        self,
        tensor: torch.Tensor,
        start: int,
        end: int,
        pad_value: float,
    ) -> torch.Tensor:
        window = tensor[..., start:end]

        if window.shape[-1] < self.snippet_frames:
            if not self.pad_short_tracks:
                raise RuntimeError(
                    f"Track shorter than snippet length: "
                    f"{window.shape[-1]} < {self.snippet_frames}"
                )
            window = pad_feature_last_dim(window, self.snippet_frames, pad_value)

        return window

    def _make_item_from_payload(
        self,
        payload: dict[str, Any],
        cache_path: Path,
        record_index: int,
        idx: int,
    ) -> dict[str, Any]:
        cqt = payload["cqt_mag"].to(dtype=self.dtype)

        total_frames = int(cqt.shape[-1])
        start = self._choose_start_frame(total_frames, idx)
        end = start + self.snippet_frames

        cqt_window = self._slice_last_dim(
            cqt,
            start=start,
            end=end,
            pad_value=-1.0,
        )

        item: dict[str, Any] = {
            "cqt": cqt_window,
            "frame_start": int(start),
            "frame_end": int(end),
            "track_index": int(record_index),
            "domain": self.domain,
            "cache_path": str(cache_path),
            "energy": cqt_window_energy(cqt_window),
        }

        if self.return_chroma and "chroma" in payload:
            chroma = payload["chroma"].to(dtype=self.dtype)
            item["chroma"] = self._slice_last_dim(
                chroma,
                start=start,
                end=end,
                pad_value=0.0,
            )

        if self.return_phase and "cqt_phase" in payload:
            phase = payload["cqt_phase"].to(dtype=self.dtype)
            item["phase"] = self._slice_last_dim(
                phase,
                start=start,
                end=end,
                pad_value=0.0,
            )

        metadata = payload.get("metadata", {})

        if self.return_metadata:
            item["metadata"] = metadata

        source_path = metadata.get("source_path")
        if source_path is not None:
            item["source_path"] = source_path

        return item

    def __getitem__(self, idx: int) -> dict[str, Any]:
        last_item: dict[str, Any] | None = None

        for attempt in range(self.max_resample_attempts):
            record_index = self._record_index_for_item(idx)

            # If the selected snippet is too quiet, retry with a random track/window.
            if attempt > 0:
                record_index = random.randint(0, len(self.records) - 1)

            payload, cache_path = self._load_payload(record_index)
            item = self._make_item_from_payload(payload, cache_path, record_index, idx)

            last_item = item

            if item["energy"] >= self.min_window_energy:
                return item

        # Fallback: return the last sampled item rather than hard failing.
        assert last_item is not None
        return last_item


Domain = Literal["poly", "chip"]
