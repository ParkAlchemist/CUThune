from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.utils.data import (
    Dataset,
    RandomSampler,
    SequentialSampler,
)
from unittest.mock import patch

from cut.src.nes_cut.data.unpaired_dataset import (
    UnpairedCQTDataset,
    _mixed_index,
    build_unpaired_cqt_dataloader,
    build_unpaired_cqt_dataset,
    cqt_collate_fn,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_BINS = 116
SNIPPET_FRAMES = 688
SAMPLE_RATE = 44_100
HOP_LENGTH = 256


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def make_cqt_spec() -> SimpleNamespace:
    """
    Minimal CQT specification required by UnpairedCQTDataset.

    Replace SimpleNamespace with the real CQTSpec once its constructor
    and import path are finalized.
    """
    return SimpleNamespace(
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        fmin=27.5,
        bins_per_octave=12,
        n_bins=N_BINS,
        db_min=-80.0,
        db_max=0.0,
        normalized_min=-1.0,
        normalized_max=1.0,
        version="cqt-a0-116-v1",
    )


def make_track_item(
    *,
    track_id: str,
    value: float,
    frame_start: int = 0,
    num_frames: int = SNIPPET_FRAMES,
    n_bins: int = N_BINS,
    finite: bool = True,
    include_source_path: bool = True,
) -> dict[str, Any]:
    """
    Construct one CachedCQTTrackDataset-compatible item.

    A constant fill value makes selected target indices directly observable.
    """
    if finite:
        cqt = torch.full(
            (1, n_bins, num_frames),
            fill_value=value,
            dtype=torch.float32,
        )
    else:
        cqt = torch.full(
            (1, n_bins, num_frames),
            fill_value=float("inf"),
            dtype=torch.float32,
        )

    item: dict[str, Any] = {
        "cqt": cqt,
        "energy": float(abs(value)),
        "frame_start": frame_start,
        "frame_end": frame_start + num_frames,
        "cache_path": f"/cache/{track_id}.pt",
        "track_id": track_id,
    }

    if include_source_path:
        item["source_path"] = f"/audio/{track_id}.wav"

    return item


def make_domain_items(
    *,
    prefix: str,
    count: int,
) -> list[dict[str, Any]]:
    return [
        make_track_item(
            track_id=f"{prefix}-{index:03d}",
            value=float(index + 1),
            frame_start=index * SNIPPET_FRAMES,
        )
        for index in range(count)
    ]


class FakeCQTTrackDataset(Dataset[dict[str, Any]]):
    """
    Index-aware replacement for CachedCQTTrackDataset.

    The accessed_indices list allows tests to verify pairing behavior directly.
    """

    def __init__(
        self,
        items: list[dict[str, Any]],
        *,
        cqt_spec: Any | None = None,
    ) -> None:
        self.items = items
        self.cqt_spec = cqt_spec
        self.accessed_indices: list[int] = []

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        self.accessed_indices.append(index)
        return self.items[index]

    @property
    def track_ids(self) -> tuple[str, ...]:
        return tuple(
            str(item["track_id"])
            for item in self.items
        )


class FakeUnpairedDataset(Dataset[dict[str, Any]]):
    """
    Minimal dataset used for DataLoader configuration tests.
    """

    def __init__(
        self,
        *,
        length: int = 5,
    ) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        return make_cut_batch_item(
            source_id=f"source-{index}",
            target_id=f"target-{index}",
            source_value=float(index),
            target_value=float(index + 100),
        )


def make_cut_batch_item(
    *,
    source_id: str,
    target_id: str,
    source_value: float,
    target_value: float,
) -> dict[str, Any]:
    """
    Construct the stable CUT item schema returned by UnpairedCQTDataset.

    Valid masks are intentionally omitted. Add them here only if they are
    implemented as part of the Stage 1 dataset contract.
    """
    return {
        "source_cqt": torch.full(
            (1, N_BINS, SNIPPET_FRAMES),
            source_value,
            dtype=torch.float32,
        ),
        "target_cqt": torch.full(
            (1, N_BINS, SNIPPET_FRAMES),
            target_value,
            dtype=torch.float32,
        ),
        "source_energy": abs(source_value),
        "target_energy": abs(target_value),
        "source_track_id": source_id,
        "target_track_id": target_id,
        "source_frame_start": 0,
        "source_frame_end": SNIPPET_FRAMES,
        "target_frame_start": 100,
        "target_frame_end": 100 + SNIPPET_FRAMES,
        "source_cache_path": f"/cache/{source_id}.pt",
        "target_cache_path": f"/cache/{target_id}.pt",
        "source_path": f"/audio/{source_id}.wav",
        "target_path": f"/audio/{target_id}.wav",
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cqt_spec() -> SimpleNamespace:
    return make_cqt_spec()


@pytest.fixture
def source_dataset(
    cqt_spec: SimpleNamespace,
) -> FakeCQTTrackDataset:
    return FakeCQTTrackDataset(
        make_domain_items(
            prefix="source",
            count=10,
        ),
        cqt_spec=cqt_spec,
    )


@pytest.fixture
def target_dataset(
    cqt_spec: SimpleNamespace,
) -> FakeCQTTrackDataset:
    return FakeCQTTrackDataset(
        make_domain_items(
            prefix="target",
            count=20,
        ),
        cqt_spec=cqt_spec,
    )


@pytest.fixture
def unpaired_dataset(
    source_dataset: FakeCQTTrackDataset,
    target_dataset: FakeCQTTrackDataset,
    cqt_spec: SimpleNamespace,
) -> UnpairedCQTDataset:
    return UnpairedCQTDataset(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        cqt_spec=cqt_spec,
        snippet_frames=SNIPPET_FRAMES,
        length_mode="max",
        random_target_pairing=True,
        pairing_seed=1337,
    )


@pytest.fixture
def batch() -> list[dict[str, Any]]:
    return [
        make_cut_batch_item(
            source_id="source-001",
            target_id="target-001",
            source_value=1.0,
            target_value=101.0,
        ),
        make_cut_batch_item(
            source_id="source-002",
            target_id="target-002",
            source_value=2.0,
            target_value=102.0,
        ),
    ]


# ---------------------------------------------------------------------------
# Dataset schema and geometry
# ---------------------------------------------------------------------------


def test_unpaired_dataset_returns_cut_schema(
    unpaired_dataset: UnpairedCQTDataset,
) -> None:
    item = unpaired_dataset[0]

    expected_keys = {
        "source_cqt",
        "target_cqt",
        "source_energy",
        "target_energy",
        "source_track_id",
        "target_track_id",
        "source_frame_start",
        "source_frame_end",
        "target_frame_start",
        "target_frame_end",
        "source_cache_path",
        "target_cache_path",
        "source_path",
        "target_path",
        "source_valid_mask",
        "target_valid_mask",
    }

    assert set(item) == expected_keys


def test_unpaired_dataset_returns_116_by_688_cqt(
    unpaired_dataset: UnpairedCQTDataset,
    cqt_spec: SimpleNamespace,
) -> None:
    item = unpaired_dataset[0]

    expected_shape = (
        1,
        cqt_spec.n_bins,
        SNIPPET_FRAMES,
    )

    assert expected_shape == (1, 116, 688)
    assert item["source_cqt"].shape == expected_shape
    assert item["target_cqt"].shape == expected_shape


def test_unpaired_dataset_returns_expected_metadata(
    unpaired_dataset: UnpairedCQTDataset,
) -> None:
    item = unpaired_dataset[0]

    assert item["source_track_id"] == "source-000"
    assert item["source_frame_start"] == 0
    assert item["source_frame_end"] == SNIPPET_FRAMES
    assert item["source_cache_path"].endswith(
        "source-000.pt"
    )
    assert item["source_path"].endswith(
        "source-000.wav"
    )

    assert item["target_track_id"].startswith(
        "target-"
    )
    assert item["target_cache_path"].endswith(
        ".pt"
    )


def test_optional_source_path_is_returned_as_none(
    target_dataset: FakeCQTTrackDataset,
    cqt_spec: SimpleNamespace,
) -> None:
    source = FakeCQTTrackDataset(
        [
            make_track_item(
                track_id="source-no-path",
                value=1.0,
                include_source_path=False,
            )
        ],
        cqt_spec=cqt_spec,
    )

    dataset = UnpairedCQTDataset(
        source_dataset=source,
        target_dataset=target_dataset,
        cqt_spec=cqt_spec,
        snippet_frames=SNIPPET_FRAMES,
        random_target_pairing=False,
    )

    item = dataset[0]

    assert item["source_path"] is None


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_unpaired_dataset_rejects_empty_source(
    target_dataset: FakeCQTTrackDataset,
    cqt_spec: SimpleNamespace,
) -> None:
    empty_source = FakeCQTTrackDataset(
        [],
        cqt_spec=cqt_spec,
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        UnpairedCQTDataset(
            source_dataset=empty_source,
            target_dataset=target_dataset,
            cqt_spec=cqt_spec,
            snippet_frames=SNIPPET_FRAMES,
        )


def test_unpaired_dataset_rejects_empty_target(
    source_dataset: FakeCQTTrackDataset,
    cqt_spec: SimpleNamespace,
) -> None:
    empty_target = FakeCQTTrackDataset(
        [],
        cqt_spec=cqt_spec,
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        UnpairedCQTDataset(
            source_dataset=source_dataset,
            target_dataset=empty_target,
            cqt_spec=cqt_spec,
            snippet_frames=SNIPPET_FRAMES,
        )


def test_unpaired_dataset_rejects_nonpositive_frames(
    source_dataset: FakeCQTTrackDataset,
    target_dataset: FakeCQTTrackDataset,
    cqt_spec: SimpleNamespace,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be greater than 0",
    ):
        UnpairedCQTDataset(
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            cqt_spec=cqt_spec,
            snippet_frames=0,
        )


# ---------------------------------------------------------------------------
# CQT validation
# ---------------------------------------------------------------------------


def test_unpaired_dataset_rejects_nonfinite_source_cqt(
    target_dataset: FakeCQTTrackDataset,
    cqt_spec: SimpleNamespace,
) -> None:
    source = FakeCQTTrackDataset(
        [
            make_track_item(
                track_id="source-inf",
                value=1.0,
                finite=False,
            )
        ],
        cqt_spec=cqt_spec,
    )

    dataset = UnpairedCQTDataset(
        source_dataset=source,
        target_dataset=target_dataset,
        cqt_spec=cqt_spec,
        snippet_frames=SNIPPET_FRAMES,
        random_target_pairing=False,
    )

    with pytest.raises(
        ValueError,
        match="source CQT has nan or inf",
    ):
        dataset[0]


def test_unpaired_dataset_rejects_nonfinite_target_cqt(
    source_dataset: FakeCQTTrackDataset,
    cqt_spec: SimpleNamespace,
) -> None:
    target = FakeCQTTrackDataset(
        [
            make_track_item(
                track_id="target-inf",
                value=1.0,
                finite=False,
            )
        ],
        cqt_spec=cqt_spec,
    )

    dataset = UnpairedCQTDataset(
        source_dataset=source_dataset,
        target_dataset=target,
        cqt_spec=cqt_spec,
        snippet_frames=SNIPPET_FRAMES,
        random_target_pairing=False,
    )

    with pytest.raises(
        ValueError,
        match="target CQT has nan or inf",
    ):
        dataset[0]


def test_unpaired_dataset_rejects_wrong_source_cqt_shape(
    target_dataset: FakeCQTTrackDataset,
    cqt_spec: SimpleNamespace,
) -> None:
    source = FakeCQTTrackDataset(
        [
            make_track_item(
                track_id="source-wrong-shape",
                value=1.0,
                num_frames=689,
            )
        ],
        cqt_spec=cqt_spec,
    )

    dataset = UnpairedCQTDataset(
        source_dataset=source,
        target_dataset=target_dataset,
        cqt_spec=cqt_spec,
        snippet_frames=SNIPPET_FRAMES,
        random_target_pairing=False,
    )

    with pytest.raises(
        ValueError,
        match="source CQT has shape",
    ):
        dataset[0]


def test_unpaired_dataset_rejects_wrong_target_cqt_shape(
    source_dataset: FakeCQTTrackDataset,
    cqt_spec: SimpleNamespace,
) -> None:
    target = FakeCQTTrackDataset(
        [
            make_track_item(
                track_id="target-wrong-shape",
                value=1.0,
                n_bins=115,
            )
        ],
        cqt_spec=cqt_spec,
    )

    dataset = UnpairedCQTDataset(
        source_dataset=source_dataset,
        target_dataset=target,
        cqt_spec=cqt_spec,
        snippet_frames=SNIPPET_FRAMES,
        random_target_pairing=False,
    )

    with pytest.raises(
        ValueError,
        match="target CQT has shape",
    ):
        dataset[0]


# ---------------------------------------------------------------------------
# Dataset length modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("length_mode", "expected"),
    [
        ("max", 20),
        ("min", 10),
        ("source", 10),
        ("target", 20),
    ],
)
def test_length_calculation(
    unpaired_dataset: UnpairedCQTDataset,
    length_mode: str,
    expected: int,
) -> None:
    unpaired_dataset.length_mode = length_mode

    assert len(unpaired_dataset) == expected


def test_length_calculation_rejects_unknown_mode(
    unpaired_dataset: UnpairedCQTDataset,
) -> None:
    unpaired_dataset.length_mode = "invalid"

    with pytest.raises(
        ValueError,
        match="Unsupported length mode",
    ):
        len(unpaired_dataset)


# ---------------------------------------------------------------------------
# Pairing behavior
# ---------------------------------------------------------------------------


def test_training_target_pairing_is_reproducible(
    unpaired_dataset: UnpairedCQTDataset,
    target_dataset: FakeCQTTrackDataset,
) -> None:
    source_index = 4

    first_item = unpaired_dataset[source_index]
    first_target_index = target_dataset.accessed_indices[-1]

    second_item = unpaired_dataset[source_index]
    second_target_index = target_dataset.accessed_indices[-1]

    assert first_target_index == second_target_index
    assert (
        first_item["target_track_id"]
        == second_item["target_track_id"]
    )
    assert torch.equal(
        first_item["target_cqt"],
        second_item["target_cqt"],
    )


def test_training_pairing_uses_mixed_index(
    unpaired_dataset: UnpairedCQTDataset,
    target_dataset: FakeCQTTrackDataset,
) -> None:
    source_index = 7

    expected_target_index = _mixed_index(
        seed=unpaired_dataset.pairing_seed,
        index=source_index,
        modulus=len(target_dataset),
    )

    item = unpaired_dataset[source_index]

    assert (
        target_dataset.accessed_indices[-1]
        == expected_target_index
    )
    assert item["target_track_id"] == (
        f"target-{expected_target_index:03d}"
    )


def test_different_pairing_seeds_change_pairing_sequence() -> None:
    first_sequence = [
        _mixed_index(
            seed=1337,
            index=index,
            modulus=20,
        )
        for index in range(20)
    ]

    second_sequence = [
        _mixed_index(
            seed=100,
            index=index,
            modulus=20,
        )
        for index in range(20)
    ]

    assert first_sequence != second_sequence


def test_validation_pairing_uses_index_modulo_target_length(
    source_dataset: FakeCQTTrackDataset,
    target_dataset: FakeCQTTrackDataset,
    cqt_spec: SimpleNamespace,
) -> None:
    dataset = UnpairedCQTDataset(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        cqt_spec=cqt_spec,
        snippet_frames=SNIPPET_FRAMES,
        length_mode="min",
        random_target_pairing=False,
    )

    source_index = 7
    item = dataset[source_index]

    assert target_dataset.accessed_indices[-1] == 7
    assert item["target_track_id"] == "target-007"


def test_validation_pairing_wraps_target_index(
    source_dataset: FakeCQTTrackDataset,
    target_dataset: FakeCQTTrackDataset,
    cqt_spec: SimpleNamespace,
) -> None:
    dataset = UnpairedCQTDataset(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        cqt_spec=cqt_spec,
        snippet_frames=SNIPPET_FRAMES,
        length_mode="max",
        random_target_pairing=False,
    )

    source_index = 23

    # Source wraps to index 3 because the source contains 10 items.
    # Target wraps to index 3 because the target contains 20 items.
    item = dataset[source_index]

    assert source_dataset.accessed_indices[-1] == 3
    assert target_dataset.accessed_indices[-1] == 3
    assert item["source_track_id"] == "source-003"
    assert item["target_track_id"] == "target-003"


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def test_collate_preserves_track_ids_as_lists(
    batch: list[dict[str, Any]],
) -> None:
    collated = cqt_collate_fn(batch)

    assert collated["source_track_id"] == [
        "source-001",
        "source-002",
    ]
    assert collated["target_track_id"] == [
        "target-001",
        "target-002",
    ]


def test_collate_preserves_paths_as_lists(
    batch: list[dict[str, Any]],
) -> None:
    collated = cqt_collate_fn(batch)

    assert isinstance(
        collated["source_cache_path"],
        list,
    )
    assert isinstance(
        collated["target_cache_path"],
        list,
    )
    assert isinstance(
        collated["source_path"],
        list,
    )
    assert isinstance(
        collated["target_path"],
        list,
    )


def test_collate_stacks_cqt_tensors(
    batch: list[dict[str, Any]],
) -> None:
    collated = cqt_collate_fn(batch)

    assert collated["source_cqt"].shape == (
        2,
        1,
        N_BINS,
        SNIPPET_FRAMES,
    )
    assert collated["target_cqt"].shape == (
        2,
        1,
        N_BINS,
        SNIPPET_FRAMES,
    )

    assert torch.equal(
        collated["source_cqt"][0],
        batch[0]["source_cqt"],
    )
    assert torch.equal(
        collated["source_cqt"][1],
        batch[1]["source_cqt"],
    )


def test_collate_stacks_numeric_values(
    batch: list[dict[str, Any]],
) -> None:
    collated = cqt_collate_fn(batch)

    assert collated["source_energy"].shape == (2,)
    assert collated["target_energy"].shape == (2,)
    assert collated["source_frame_start"].shape == (2,)
    assert collated["source_frame_end"].shape == (2,)


def test_collate_rejects_inconsistent_schema(
    batch: list[dict[str, Any]],
) -> None:
    malformed_batch = deepcopy(batch)
    malformed_batch[0]["extra_key"] = "value"

    with pytest.raises(
        ValueError,
        match="Inconsistent sample schema",
    ):
        cqt_collate_fn(malformed_batch)


def test_collate_rejects_missing_key(
    batch: list[dict[str, Any]],
) -> None:
    malformed_batch = deepcopy(batch)
    del malformed_batch[1]["target_energy"]

    with pytest.raises(
        ValueError,
        match="Inconsistent sample schema",
    ):
        cqt_collate_fn(malformed_batch)


def test_collate_rejects_empty_batch() -> None:
    with pytest.raises(
        ValueError,
        match="empty batch",
    ):
        cqt_collate_fn([])


# ---------------------------------------------------------------------------
# DataLoader configuration
# ---------------------------------------------------------------------------


@patch(
    "cut.src.nes_cut.data.unpaired_dataset."
    "build_unpaired_cqt_dataset"
)
def test_training_loader_configuration(
    build_dataset_mock,
    cqt_spec: SimpleNamespace,
) -> None:
    build_dataset_mock.return_value = (
        FakeUnpairedDataset(length=5)
    )

    loader = build_unpaired_cqt_dataloader(
        Path("/unused"),
        cqt_spec=cqt_spec,
        batch_size=2,
        num_workers=0,
        snippet_frames=SNIPPET_FRAMES,
        split="train",
        pin_memory=False,
        persistent_workers=False,
    )

    assert loader.drop_last
    assert isinstance(
        loader.sampler,
        RandomSampler,
    )
    assert not loader.persistent_workers


@patch(
    "cut.src.nes_cut.data.unpaired_dataset."
    "build_unpaired_cqt_dataset"
)
def test_validation_loader_configuration(
    build_dataset_mock,
    cqt_spec: SimpleNamespace,
) -> None:
    build_dataset_mock.return_value = (
        FakeUnpairedDataset(length=5)
    )

    loader = build_unpaired_cqt_dataloader(
        Path("/unused"),
        cqt_spec=cqt_spec,
        batch_size=2,
        num_workers=0,
        snippet_frames=SNIPPET_FRAMES,
        split="val",
        pin_memory=False,
        persistent_workers=False,
    )

    assert not loader.drop_last
    assert isinstance(
        loader.sampler,
        SequentialSampler,
    )
    assert not loader.persistent_workers


@patch(
    "cut.src.nes_cut.data.unpaired_dataset."
    "build_unpaired_cqt_dataset"
)
def test_loader_disables_persistent_workers_when_worker_count_is_zero(
    build_dataset_mock,
    cqt_spec: SimpleNamespace,
) -> None:
    build_dataset_mock.return_value = (
        FakeUnpairedDataset(length=5)
    )

    loader = build_unpaired_cqt_dataloader(
        Path("/unused"),
        cqt_spec=cqt_spec,
        batch_size=2,
        num_workers=0,
        split="train",
        persistent_workers=True,
    )

    assert not loader.persistent_workers


# ---------------------------------------------------------------------------
# Builder configuration
# ---------------------------------------------------------------------------


@patch(
    "cut.src.nes_cut.data.unpaired_dataset."
    "CachedCQTTrackDataset"
)
def test_builder_constructs_both_domains_from_shared_cqt_spec(
    dataset_class_mock,
    cqt_spec: SimpleNamespace,
) -> None:
    source = FakeCQTTrackDataset(
        make_domain_items(
            prefix="source",
            count=2,
        ),
        cqt_spec=cqt_spec,
    )
    target = FakeCQTTrackDataset(
        make_domain_items(
            prefix="target",
            count=2,
        ),
        cqt_spec=cqt_spec,
    )

    dataset_class_mock.side_effect = [
        source,
        target,
    ]

    dataset = build_unpaired_cqt_dataset(
        Path("/cache"),
        cqt_spec=cqt_spec,
        snippet_frames=SNIPPET_FRAMES,
        windows_per_track=16,
        split="train",
        pairing_seed=1337,
    )

    assert isinstance(
        dataset,
        UnpairedCQTDataset,
    )
    assert dataset_class_mock.call_count == 2

    source_call = (
        dataset_class_mock.call_args_list[0]
    )
    target_call = (
        dataset_class_mock.call_args_list[1]
    )

    assert source_call.kwargs["cache_root"] == (
        Path("/cache") / "poly"
    )
    assert target_call.kwargs["cache_root"] == (
        Path("/cache") / "chip"
    )

    assert source_call.kwargs["domain"] == "poly"
    assert target_call.kwargs["domain"] == "chip"

    for constructor_call in (
        source_call,
        target_call,
    ):
        assert (
            constructor_call.kwargs["sample_rate"]
            == cqt_spec.sample_rate
        )
        assert (
            constructor_call.kwargs["hop_length"]
            == cqt_spec.hop_length
        )
        assert (
            constructor_call.kwargs["snippet_frames"]
            == SNIPPET_FRAMES
        )
        assert (
            constructor_call.kwargs["return_chroma"]
            is False
        )
        assert (
            constructor_call.kwargs["return_phase"]
            is False
        )
        assert (
            constructor_call.kwargs["return_metadata"]
            is True
        )


@patch(
    "cut.src.nes_cut.data.unpaired_dataset."
    "CachedCQTTrackDataset"
)
def test_training_builder_enables_random_windows_and_pairing(
    dataset_class_mock,
    cqt_spec: SimpleNamespace,
) -> None:
    dataset_class_mock.side_effect = [
        FakeCQTTrackDataset(
            make_domain_items(
                prefix="source",
                count=2,
            ),
            cqt_spec=cqt_spec,
        ),
        FakeCQTTrackDataset(
            make_domain_items(
                prefix="target",
                count=2,
            ),
            cqt_spec=cqt_spec,
        ),
    ]

    dataset = build_unpaired_cqt_dataset(
        Path("/cache"),
        cqt_spec=cqt_spec,
        split="train",
    )

    for constructor_call in (
        dataset_class_mock.call_args_list
    ):
        assert (
            constructor_call.kwargs["random_window"]
            is True
        )
        assert (
            constructor_call.kwargs["random_track"]
            is False
        )

    assert dataset.random_target_pairing is True


@patch(
    "cut.src.nes_cut.data.unpaired_dataset."
    "CachedCQTTrackDataset"
)
def test_validation_builder_is_deterministic(
    dataset_class_mock,
    cqt_spec: SimpleNamespace,
) -> None:
    dataset_class_mock.side_effect = [
        FakeCQTTrackDataset(
            make_domain_items(
                prefix="source",
                count=2,
            ),
            cqt_spec=cqt_spec,
        ),
        FakeCQTTrackDataset(
            make_domain_items(
                prefix="target",
                count=2,
            ),
            cqt_spec=cqt_spec,
        ),
    ]

    dataset = build_unpaired_cqt_dataset(
        Path("/cache"),
        cqt_spec=cqt_spec,
        split="val",
    )

    for constructor_call in (
        dataset_class_mock.call_args_list
    ):
        assert (
            constructor_call.kwargs["random_window"]
            is False
        )
        assert (
            constructor_call.kwargs["random_track"]
            is False
        )

    assert dataset.random_target_pairing is False

