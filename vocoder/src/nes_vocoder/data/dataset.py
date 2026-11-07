from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.cqt_dataset import discover_cache_records
from src.data.cqt_feature_utils import (
    load_audio,
    trim_silence,
    peak_normalize,
)


class WaveformLRUCache:
    def __init__(self, max_items: int = 8) -> None:
        self.max_items = max(0, int(max_items))
        self.items: OrderedDict[str, torch.Tensor] = OrderedDict()

    def get(self, key: str) -> torch.Tensor | None:
        if self.max_items <= 0:
            return None

        if key not in self.items:
            return None

        value = self.items.pop(key)
        self.items[key] = value
        return value

    def put(self, key: str, value: torch.Tensor) -> None:
        if self.max_items <= 0:
            return

        self.items[key] = value

        while len(self.items) > self.max_items:
            self.items.popitem(last=False)


def pad_1d(x: torch.Tensor, target_length: int) -> torch.Tensor:
    if x.shape[-1] >= target_length:
        return x[..., :target_length]

    pad = target_length - x.shape[-1]
    return torch.nn.functional.pad(x, (0, pad))


def load_processed_waveform(
        source_path: Path,
        sample_rate: int,
        trim_top_db: float = 60.0,
        target_peak: float = 0.95,
) -> torch.Tensor:
    """
    Recreate the waveform preprocessing used for CQT cache generation.

    Long-term improvement:
        Store processed waveform or trim indices during preprocessing. For now,
        this mirrors the existing CQT preprocessing assumptions closely enough
        for the first vocoder baseline.
    """
    y, _ = load_audio(
        source_path,
        sample_rate=sample_rate,
        mono=True,
    )

    y = trim_silence(y, top_db=trim_top_db)
    y = peak_normalize(y, target_peak=target_peak)

    return torch.from_numpy(y.astype(np.float32))


class CQTVocoderDataset(Dataset):
    """
    Paired dataset for vocoder training.

    Source:
        cached real-chip CQT from cache/cqt/chip/*.pt

    Target:
        corresponding processed waveform loaded from metadata source_path

    Returned item:
        {
            "cqt":   Tensor [96, segment_frames],
            "audio": Tensor [1, segment_frames * hop_length],
            ...
        }

    Alignment rule:
        frame_start k maps to sample_start k * hop_length

    Maintaining this alignment is critical for vocoder training.
    """

    def __init__(
            self,
            chip_cache_root: Path,
            sample_rate: int = 22050,
            hop_length: int = 512,
            segment_frames: int = 32,
            windows_per_track: int = 8,
            trim_top_db: float = 60.0,
            target_peak: float = 0.95,
            cache_waveforms: int = 8,
            random_window: bool = True,
    ) -> None:
        self.chip_cache_root = Path(chip_cache_root)
        self.records = discover_cache_records(self.chip_cache_root)

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.segment_frames = segment_frames
        self.segment_samples = segment_frames * hop_length
        self.windows_per_track = max(1, int(windows_per_track))

        self.trim_top_db = trim_top_db
        self.target_peak = target_peak
        self.random_window = random_window

        self.wave_cache = WaveformLRUCache(max_items=cache_waveforms)

        if len(self.records) == 0:
            raise RuntimeError(
                f"No usable CQT records found in {chip_cache_root}")

    def __len__(self) -> int:
        return len(self.records) * self.windows_per_track

    def _record_index(self, index: int) -> int:
        return index % len(self.records)

    def _load_payload(self, record_index: int) -> tuple[dict[str, Any], Path]:
        record = self.records[record_index]
        cache_path = Path(record["cache_path"])
        payload = torch.load(cache_path, map_location="cpu")

        if "cqt_mag" not in payload:
            raise KeyError(f"Missing cqt_mag in {cache_path}")

        return payload, cache_path

    def _load_waveform(self, metadata: dict[str, Any]) -> torch.Tensor:
        source_path = metadata.get("source_path")
        if source_path is None:
            raise KeyError("Cache metadata missing source_path")

        source_path = Path(source_path)
        key = str(source_path)

        cached = self.wave_cache.get(key)
        if cached is not None:
            return cached

        y = load_processed_waveform(
            source_path=source_path,
            sample_rate=self.sample_rate,
            trim_top_db=self.trim_top_db,
            target_peak=self.target_peak,
        )

        self.wave_cache.put(key, y)
        return y

    def _choose_start_frame(self, total_frames: int, index: int) -> int:
        if total_frames <= self.segment_frames:
            return 0

        max_start = total_frames - self.segment_frames

        if self.random_window:
            return int(torch.randint(0, max_start + 1, size=(1,)).item())

        virtual_idx = index // len(self.records)

        if self.windows_per_track <= 1:
            return 0

        fraction = virtual_idx / max(1, self.windows_per_track - 1)
        return int(round(max_start * fraction))

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index = self._record_index(index)
        payload, cache_path = self._load_payload(record_index)

        metadata = payload.get("metadata", {})
        cqt = payload["cqt_mag"].float()

        # [1, F, T] -> [F, T]
        if cqt.ndim == 3 and cqt.shape[0] == 1:
            cqt = cqt[0]

        if cqt.ndim != 2:
            raise ValueError(f"Expected CQT [F,T], got {tuple(cqt.shape)}")

        total_frames = int(cqt.shape[-1])
        start_frame = self._choose_start_frame(total_frames, index)
        end_frame = start_frame + self.segment_frames

        cqt_segment = cqt[:, start_frame:end_frame]

        if cqt_segment.shape[-1] < self.segment_frames:
            cqt_segment = torch.nn.functional.pad(
                cqt_segment,
                (0, self.segment_frames - cqt_segment.shape[-1]),
                value=-1.0,
            )

        waveform = self._load_waveform(metadata)

        start_sample = start_frame * self.hop_length
        end_sample = start_sample + self.segment_samples

        audio_segment = waveform[start_sample:end_sample]
        audio_segment = pad_1d(audio_segment, self.segment_samples).unsqueeze(0)

        return {
            "cqt": cqt_segment,
            "audio": audio_segment,
            "cache_path": str(cache_path),
            "source_path": metadata.get("source_path", ""),
            "frame_start": start_frame,
            "frame_end": end_frame,
            "sample_start": start_sample,
            "sample_end": end_sample,
        }

