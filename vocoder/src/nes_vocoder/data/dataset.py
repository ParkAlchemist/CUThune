from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from shared.src.chiptune_shared.cqt.cache import discover_cache_records
from shared.src.chiptune_shared.cqt.features import load_audio
from vocoder.src.nes_vocoder.config.schema import VocoderWindowSamplingConfig

EXPECTED_CACHE_VERSION = 4


@dataclass(frozen=True)
class WindowCandidates:
    # Boundary frame positions, not window starts.
    active_to_silent: torch.Tensor
    silent_to_active: torch.Tensor
    # Actual valid window starts.
    silence_rich_starts: torch.Tensor


class TensorLRUCache:
    def __init__(self, max_items: int = 8) -> None:
        self.max_items = max(0, int(max_items))
        self.items: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any | None:
        if self.max_items <= 0 or key not in self.items:
            return None
        value = self.items.pop(key)
        self.items[key] = value
        return value

    def put(self, key: str, value: Any) -> None:
        if self.max_items <= 0:
            return
        if key in self.items:
            self.items.pop(key)
        self.items[key] = value
        while len(self.items) > self.max_items:
            self.items.popitem(last=False)


def pad_to_length(
    value: torch.Tensor,
    *,
    target_length: int,
    padding_value: float,
) -> torch.Tensor:
    current_length = int(value.shape[-1])
    if current_length >= target_length:
        return value[..., :target_length]
    return F.pad(
        value,
        (0, target_length - current_length),
        value=padding_value,
    )


def stable_split_key(record: dict[str, Any]) -> str:
    value = (
        record.get("id")
        or record.get("source_path")
        or record.get("cache_path")
    )
    if value is None:
        raise KeyError("Cache record lacks id, source_path, and cache_path.")
    return str(value)


def stable_split_value(
    record: dict[str, Any],
    *,
    split_seed: int,
) -> float:
    digest = hashlib.sha1(
        f"{split_seed}:{stable_split_key(record)}".encode("utf-8")
    ).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(2**64)


def split_cache_records(
    records: Sequence[dict[str, Any]],
    *,
    validation_fraction: float,
    split_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must lie strictly between zero and one."
        )

    train_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []

    for record in records:
        if stable_split_value(record, split_seed=split_seed) < validation_fraction:
            validation_records.append(record)
        else:
            train_records.append(record)

    if not train_records:
        raise RuntimeError("Track split produced no training records.")
    if not validation_records:
        raise RuntimeError("Track split produced no validation records.")

    return train_records, validation_records


def write_split_manifest(
    path: Path,
    *,
    train_records: Sequence[dict[str, Any]],
    validation_records: Sequence[dict[str, Any]],
    split_seed: int,
    validation_fraction: float,
) -> None:
    value = {
        "split_seed": split_seed,
        "validation_fraction": validation_fraction,
        "train_ids": sorted(stable_split_key(record) for record in train_records),
        "validation_ids": sorted(
            stable_split_key(record) for record in validation_records
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def stable_transition_indices(
    silent: torch.Tensor,
    *,
    minimum_state_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if silent.ndim != 1:
        raise ValueError(f"Expected silence mask [T], got {tuple(silent.shape)}.")

    total_frames = int(silent.numel())
    active_to_silent: list[int] = []
    silent_to_active: list[int] = []

    for boundary in range(minimum_state_frames, total_frames - minimum_state_frames + 1):
        left = silent[boundary - minimum_state_frames : boundary]
        right = silent[boundary : boundary + minimum_state_frames]

        if not bool(left.any()) and bool(right.all()):
            active_to_silent.append(boundary)
        elif bool(left.all()) and not bool(right.any()):
            silent_to_active.append(boundary)

    return (
        torch.tensor(active_to_silent, dtype=torch.long),
        torch.tensor(silent_to_active, dtype=torch.long),
    )


def build_window_candidates(
    frame_rms: torch.Tensor,
    *,
    segment_frames: int,
    sampling: VocoderWindowSamplingConfig,
) -> WindowCandidates:
    if segment_frames <= 0:
        raise ValueError("segment_frames must be positive.")

    if frame_rms.ndim == 2:
        if frame_rms.shape[0] != 1:
            raise ValueError(
                f"Expected frame RMS [1,T], got {tuple(frame_rms.shape)}."
            )
        frame_rms = frame_rms[0]

    if frame_rms.ndim != 1:
        raise ValueError(f"Expected frame RMS [T], got {tuple(frame_rms.shape)}.")
    if not bool(torch.isfinite(frame_rms).all()):
        raise ValueError("Frame RMS contains non-finite values.")

    total_frames = int(frame_rms.shape[-1])
    empty = torch.empty(0, dtype=torch.long)
    if total_frames == 0:
        return WindowCandidates(empty, empty, empty)

    silent = frame_rms < sampling.silence_rms_threshold

    if total_frames >= 2 * sampling.minimum_state_frames:
        active_to_silent, silent_to_active = stable_transition_indices(
            silent,
            minimum_state_frames=sampling.minimum_state_frames,
        )
    else:
        active_to_silent, silent_to_active = empty, empty

    max_start = max(0, total_frames - segment_frames)

    if total_frames < segment_frames:
        padded_silent_frames = segment_frames - total_frames
        silence_fraction = (
            int(silent.sum().item()) + padded_silent_frames
        ) / float(segment_frames)
        silence_rich_starts = (
            torch.tensor([0], dtype=torch.long)
            if silence_fraction >= sampling.silence_rich_fraction
            else empty
        )
    else:
        window_silence_fraction = F.avg_pool1d(
            silent.float().reshape(1, 1, -1),
            kernel_size=segment_frames,
            stride=1,
        )[0, 0]
        silence_rich_starts = torch.nonzero(
            window_silence_fraction >= sampling.silence_rich_fraction,
            as_tuple=False,
        ).flatten()

    return WindowCandidates(
        # Boundaries remain absolute frame positions.
        active_to_silent=active_to_silent.unique(),
        silent_to_active=silent_to_active.unique(),
        silence_rich_starts=silence_rich_starts.clamp(
            min=0,
            max=max_start,
        ).unique(),
    )


def load_processed_waveform(
    source_path: Path,
    *,
    sample_rate: int,
    metadata: dict[str, Any],
) -> torch.Tensor:
    waveform, _ = load_audio(source_path, sample_rate=sample_rate, mono=True)

    trim_interval = metadata.get("trim_interval_samples")
    if trim_interval is not None:
        if not isinstance(trim_interval, list) or len(trim_interval) != 2:
            raise ValueError(
                "Invalid trim_interval_samples in cache metadata: "
                f"{trim_interval!r}."
            )
        start, end = int(trim_interval[0]), int(trim_interval[1])
        if not 0 <= start <= end <= len(waveform):
            raise ValueError(
                f"Invalid trim interval [{start}, {end}] for waveform "
                f"length {len(waveform)}."
            )
        waveform = waveform[start:end]

    normalization_gain = float(metadata.get("normalization_gain", 1.0))
    if not math.isfinite(normalization_gain):
        raise ValueError("normalization_gain must be finite.")

    waveform = (waveform * normalization_gain).astype(np.float32)
    if not np.isfinite(waveform).all():
        raise RuntimeError("Processed waveform contains non-finite values.")

    processed_stats = metadata.get("processed_stats", {})
    expected_samples = processed_stats.get("num_samples")
    if expected_samples is not None:
        difference = abs(len(waveform) - int(expected_samples))
        if difference > 1:
            raise RuntimeError(
                "Reconstructed waveform length does not match cache metadata: "
                f"{len(waveform)} versus {expected_samples}."
            )

    return torch.from_numpy(waveform)


class CQTVocoderDataset(Dataset):
    def __init__(
        self,
        chip_cache_root: Path,
        *,
        split: str,
        validation_fraction: float = 0.10,
        split_seed: int = 1337,
        sample_rate: int = 44100,
        hop_length: int = 256,
        segment_frames: int = 128,
        windows_per_track: int = 8,
        cache_waveforms: int = 4,
        cache_candidates: int = 256,
        sampling: VocoderWindowSamplingConfig | None = None,
        split_manifest_path: Path | None = None,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError(
                f"split must be 'train' or 'validation', got {split!r}."
            )
        if sample_rate <= 0 or hop_length <= 0 or segment_frames <= 0:
            raise ValueError(
                "sample_rate, hop_length, and segment_frames must be positive."
            )

        self.chip_cache_root = Path(chip_cache_root)
        self.split = split
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.segment_frames = int(segment_frames)
        self.segment_samples = self.segment_frames * self.hop_length
        self.windows_per_track = max(1, int(windows_per_track))
        self.sampling = sampling or VocoderWindowSamplingConfig()
        self.sampling.validate()

        all_records = discover_cache_records(self.chip_cache_root)
        train_records, validation_records = split_cache_records(
            all_records,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
        )
        self.records = train_records if split == "train" else validation_records
        if not self.records:
            raise RuntimeError(
                f"No usable {split} CQT records found in {chip_cache_root}."
            )

        if split_manifest_path is not None and split == "train":
            write_split_manifest(
                split_manifest_path,
                train_records=train_records,
                validation_records=validation_records,
                split_seed=split_seed,
                validation_fraction=validation_fraction,
            )

        self.wave_cache = TensorLRUCache(max_items=cache_waveforms)
        self.candidate_cache = TensorLRUCache(max_items=cache_candidates)

        self._validate_root_manifest()

    def __len__(self) -> int:
        return len(self.records) * self.windows_per_track

    def _record_index(self, index: int) -> int:
        if index < 0 or index >= len(self):
            raise IndexError(f"Dataset index out of range: {index}.")
        return index % len(self.records)

    def _validate_root_manifest(self) -> None:
        manifest_path = self.chip_cache_root.parent / "shared_cache_manifest.json"
        if not manifest_path.exists():
            return

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("cache_version", -1)) != EXPECTED_CACHE_VERSION:
            raise RuntimeError(
                f"Expected cache version {EXPECTED_CACHE_VERSION}, got "
                f"{manifest.get('cache_version')}."
            )

        config = manifest.get("feature_config", {})
        cqt = config.get("cqt", {})
        storage = config.get("storage", {})

        if int(cqt.get("sample_rate", -1)) != self.sample_rate:
            raise RuntimeError("Dataset sample rate disagrees with cache manifest.")
        if int(cqt.get("hop_length", -1)) != self.hop_length:
            raise RuntimeError("Dataset hop length disagrees with cache manifest.")
        if not storage.get("store_frame_rms", False):
            raise RuntimeError("Cache manifest says raw frame RMS is unavailable.")
        if not storage.get("store_normalized_frame_rms", False):
            raise RuntimeError(
                "Cache manifest says normalized frame RMS is unavailable."
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index = self._record_index(index)
        payload, cache_path = self._load_payload(record_index)
        metadata = payload.get("metadata", {})

        cqt = self._prepare_cqt(payload["cqt_mag"])
        frame_rms = self._prepare_envelope(payload["frame_rms"], name="frame_rms")
        normalized_rms = self._prepare_envelope(
            payload["frame_rms_normalized"],
            name="frame_rms_normalized",
        )

        frame_counts = {
            int(cqt.shape[-1]),
            int(frame_rms.shape[-1]),
            int(normalized_rms.shape[-1]),
        }
        if len(frame_counts) != 1:
            raise RuntimeError(
                f"CQT and RMS frame counts differ in {cache_path}: "
                f"CQT={cqt.shape[-1]}, RMS={frame_rms.shape[-1]}, "
                f"normalized RMS={normalized_rms.shape[-1]}."
            )

        total_frames = int(cqt.shape[-1])
        candidates = self._window_candidates(
            cache_path=cache_path,
            frame_rms=frame_rms,
        )

        if self.split == "train":
            start_frame, requested_mode, realized_mode = self._choose_training_start(
                total_frames=total_frames,
                candidates=candidates,
            )
        else:
            start_frame, realized_mode = self._choose_validation_start(
                total_frames=total_frames,
                index=index,
                candidates=candidates,
            )
            requested_mode = realized_mode

        end_frame = start_frame + self.segment_frames

        cqt_segment = pad_to_length(
            cqt[:, start_frame:end_frame],
            target_length=self.segment_frames,
            padding_value=-1.0,
        )
        rms_segment = pad_to_length(
            frame_rms[:, start_frame:end_frame],
            target_length=self.segment_frames,
            padding_value=0.0,
        )
        normalized_rms_segment = pad_to_length(
            normalized_rms[:, start_frame:end_frame],
            target_length=self.segment_frames,
            padding_value=-1.0,
        )

        waveform = self._load_waveform(metadata)
        start_sample = start_frame * self.hop_length
        end_sample = start_sample + self.segment_samples
        audio_segment = pad_to_length(
            waveform[start_sample:end_sample],
            target_length=self.segment_samples,
            padding_value=0.0,
        ).unsqueeze(0)

        silence_fraction = float(
            (rms_segment < self.sampling.silence_rms_threshold)
            .float()
            .mean()
        )

        return {
            "cqt": cqt_segment,
            "rms_envelope": normalized_rms_segment,
            "rms_envelope_linear": rms_segment,
            "audio": audio_segment,
            "cache_path": str(cache_path),
            "source_path": metadata.get("source_path", ""),
            "track_id": metadata.get("id", stable_split_key(self.records[record_index])),
            "split": self.split,
            "sampling_requested_mode": requested_mode,
            "sampling_mode": realized_mode,
            "sampling_fallback": float(requested_mode != realized_mode),
            "silence_fraction": silence_fraction,
            "frame_start": start_frame,
            "frame_end": end_frame,
            "sample_start": start_sample,
            "sample_end": end_sample,
        }

    def _load_payload(self, record_index: int) -> tuple[dict[str, Any], Path]:
        record = self.records[record_index]
        cache_path = Path(record["cache_path"])
        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )

        if int(payload.get("cache_version", -1)) != EXPECTED_CACHE_VERSION:
            raise RuntimeError(
                f"Expected cache version {EXPECTED_CACHE_VERSION} in "
                f"{cache_path}, got {payload.get('cache_version')}."
            )

        required_keys = ("cqt_mag", "frame_rms", "frame_rms_normalized")
        missing = [key for key in required_keys if key not in payload]
        if missing:
            raise KeyError(
                f"Cache file {cache_path} is missing required tensors: {missing}."
            )
        return payload, cache_path

    def _load_waveform(self, metadata: dict[str, Any]) -> torch.Tensor:
        source_path_value = metadata.get("source_path")
        if source_path_value is None:
            raise KeyError("Cache metadata missing source_path.")

        source_path = Path(source_path_value)
        key = (
            f"{source_path}:"
            f"{metadata.get('trim_interval_samples')}:"
            f"{metadata.get('normalization_gain')}"
        )
        cached = self.wave_cache.get(key)
        if cached is not None:
            return cached

        waveform = load_processed_waveform(
            source_path,
            sample_rate=self.sample_rate,
            metadata=metadata,
        )
        self.wave_cache.put(key, waveform)
        return waveform

    def _window_candidates(
        self,
        *,
        cache_path: Path,
        frame_rms: torch.Tensor,
    ) -> WindowCandidates:
        key = str(cache_path)
        cached = self.candidate_cache.get(key)
        if cached is not None:
            return cached

        candidates = build_window_candidates(
            frame_rms,
            segment_frames=self.segment_frames,
            sampling=self.sampling,
        )
        self.candidate_cache.put(key, candidates)
        return candidates

    @staticmethod
    def _random_integer(low: int, high_inclusive: int) -> int:
        if high_inclusive <= low:
            return low
        return int(torch.randint(low, high_inclusive + 1, size=(1,)).item())

    def _choose_candidate(self, candidates: torch.Tensor) -> int | None:
        if candidates.numel() == 0:
            return None
        index = self._random_integer(0, int(candidates.numel()) - 1)
        return int(candidates[index].item())

    def _center_around_boundary(self, *, boundary_frame: int, max_start: int) -> int:
        centered_start = boundary_frame - self.segment_frames // 2
        jitter = self._random_integer(
            -self.sampling.boundary_jitter_frames,
            self.sampling.boundary_jitter_frames,
        )
        return max(0, min(max_start, centered_start + jitter))

    def _choose_training_start(
        self,
        *,
        total_frames: int,
        candidates: WindowCandidates,
    ) -> tuple[int, str, str]:
        max_start = max(0, total_frames - self.segment_frames)
        draw = float(torch.rand(()).item())

        active_to_silent_end = self.sampling.active_to_silent_probability
        silent_to_active_end = (
            active_to_silent_end + self.sampling.silent_to_active_probability
        )
        silence_rich_end = (
            silent_to_active_end + self.sampling.silence_rich_probability
        )

        if draw < active_to_silent_end:
            requested = "active_to_silent"
            boundary = self._choose_candidate(candidates.active_to_silent)
            if boundary is not None:
                return (
                    self._center_around_boundary(
                        boundary_frame=boundary,
                        max_start=max_start,
                    ),
                    requested,
                    requested,
                )

        elif draw < silent_to_active_end:
            requested = "silent_to_active"
            boundary = self._choose_candidate(candidates.silent_to_active)
            if boundary is not None:
                return (
                    self._center_around_boundary(
                        boundary_frame=boundary,
                        max_start=max_start,
                    ),
                    requested,
                    requested,
                )

        elif draw < silence_rich_end:
            requested = "silence_rich"
            start = self._choose_candidate(candidates.silence_rich_starts)
            if start is not None:
                return start, requested, requested

        else:
            requested = "random"

        return self._random_integer(0, max_start), requested, "random"

    def _choose_validation_start(
        self,
        *,
        total_frames: int,
        index: int,
        candidates: WindowCandidates,
    ) -> tuple[int, str]:
        max_start = max(0, total_frames - self.segment_frames)
        virtual_index = index // len(self.records)
        mode = virtual_index % self.windows_per_track

        if mode == 0:
            return 0, "validation_start"
        if mode == 1:
            return max_start // 2, "validation_center"
        if mode == 2:
            return max_start, "validation_end"

        if mode == 3 and candidates.silence_rich_starts.numel() > 0:
            middle = candidates.silence_rich_starts.numel() // 2
            return (
                int(candidates.silence_rich_starts[middle].item()),
                "validation_silence_rich",
            )

        if mode == 4:
            all_boundaries = torch.cat(
                [candidates.active_to_silent, candidates.silent_to_active]
            ).sort().values
            if all_boundaries.numel() > 0:
                boundary = int(all_boundaries[all_boundaries.numel() // 2].item())
                return (
                    max(
                        0,
                        min(max_start, boundary - self.segment_frames // 2),
                    ),
                    "validation_boundary",
                )

        if self.windows_per_track <= 1:
            return 0, "validation_start"

        fraction = virtual_index / max(1, self.windows_per_track - 1)
        return int(round(max_start * fraction)), "validation_uniform"

    @staticmethod
    def _prepare_cqt(value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 2:
            raise ValueError(f"Expected CQT [F,T], got {tuple(value.shape)}.")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("CQT contains non-finite values.")
        return value

    @staticmethod
    def _prepare_envelope(value: torch.Tensor, *, name: str) -> torch.Tensor:
        value = value.float()
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[0] != 1:
            raise ValueError(f"Expected {name} [1,T], got {tuple(value.shape)}.")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains non-finite values.")
        return value


def build_vocoder_datasets(
    *,
    chip_cache_root: Path,
    sample_rate: int,
    hop_length: int,
    segment_frames: int,
    train_windows_per_track: int,
    validation_windows_per_track: int,
    validation_fraction: float,
    split_seed: int,
    cache_waveforms: int,
    sampling: VocoderWindowSamplingConfig,
    split_manifest_path: Path | None = None,
) -> tuple[CQTVocoderDataset, CQTVocoderDataset]:
    train_dataset = CQTVocoderDataset(
        chip_cache_root,
        split="train",
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        sample_rate=sample_rate,
        hop_length=hop_length,
        segment_frames=segment_frames,
        windows_per_track=train_windows_per_track,
        cache_waveforms=cache_waveforms,
        sampling=sampling,
        split_manifest_path=split_manifest_path,
    )

    validation_dataset = CQTVocoderDataset(
        chip_cache_root,
        split="validation",
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        sample_rate=sample_rate,
        hop_length=hop_length,
        segment_frames=segment_frames,
        windows_per_track=validation_windows_per_track,
        cache_waveforms=max(2, cache_waveforms // 2),
        sampling=sampling,
    )

    train_ids = {stable_split_key(record) for record in train_dataset.records}
    validation_ids = {
        stable_split_key(record) for record in validation_dataset.records
    }
    overlap = train_ids & validation_ids
    if overlap:
        raise RuntimeError(
            "Track leakage detected between training and validation: "
            f"{len(overlap)} records."
        )

    return train_dataset, validation_dataset


