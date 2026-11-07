import torch

from tests.test_config import (
    CACHE_ROOT,
    SAMPLE_RATE,
    HOP_LENGTH,
    N_BINS,
    SNIPPET_SECONDS,
    BATCH_SIZE,
)

from cut.src.nes_cut.data.unpaired_dataset import build_unpaired_cqt_dataset, build_unpaired_cqt_dataloader


def test_unpaired_cqt_dataset_can_be_constructed():
    dataset = build_unpaired_cqt_dataset(
        cache_root=CACHE_ROOT,
        snippet_seconds=SNIPPET_SECONDS,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        windows_per_track=2,
        return_chroma=True,
        return_phase=False,
        return_metadata=False,
        min_window_energy=0.01,
    )

    assert len(dataset) > 0


def test_unpaired_cqt_dataset_single_item_shapes():
    dataset = build_unpaired_cqt_dataset(
        cache_root=CACHE_ROOT,
        snippet_seconds=SNIPPET_SECONDS,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        windows_per_track=2,
        return_chroma=True,
        return_phase=True,
        return_metadata=False,
        min_window_energy=0.01,
    )

    item = dataset[0]

    assert "real_x" in item
    assert "real_y" in item

    assert item["real_x"].ndim == 3
    assert item["real_y"].ndim == 3

    assert item["real_x"].shape[0] == 1
    assert item["real_y"].shape[0] == 1

    assert item["real_x"].shape[1] == N_BINS
    assert item["real_y"].shape[1] == N_BINS

    assert item["real_x"].shape[-1] == item["real_y"].shape[-1]


    if "chroma_x" in item and "chroma_y" in item:
        assert item["chroma_x"].shape[0] == 12
        assert item["chroma_y"].shape[0] == 12

    if "phase_x" in item and "phase_y" in item:
        assert item["phase_x"].shape[0] == N_BINS
        assert item["phase_y"].shape[0] == N_BINS


def test_unpaired_cqt_dataloader_batch_shapes():
    loader = build_unpaired_cqt_dataloader(
        cache_root=CACHE_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=True,
        snippet_seconds=SNIPPET_SECONDS,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        windows_per_track=2,
        return_chroma=True,
        return_phase=True,
        return_metadata=False,
        min_window_energy=0.01,
    )

    batch = next(iter(loader))

    assert batch["real_x"].ndim == 4
    assert batch["real_y"].ndim == 4

    assert batch["real_x"].shape[0] == BATCH_SIZE
    assert batch["real_y"].shape[0] == BATCH_SIZE

    assert batch["real_x"].shape[1] == 1
    assert batch["real_y"].shape[1] == 1

    assert batch["real_x"].shape[2] == N_BINS
    assert batch["real_y"].shape[2] == N_BINS

    assert batch["real_x"].dtype == torch.float32
    assert batch["real_y"].dtype == torch.float32

    assert batch["real_x"].min() >= -1.05
    assert batch["real_x"].max() <= 1.05
    assert batch["real_y"].min() >= -1.05
    assert batch["real_y"].max() <= 1.05

    if "chroma_x" in batch and "chroma_y" in batch:
        assert batch["chroma_x"].shape[0] == BATCH_SIZE
        assert batch["chroma_y"].shape[0] == BATCH_SIZE

    if "phase_x" in batch and "phase_y" in batch:
        assert batch["phase_x"].shape[0] == BATCH_SIZE
        assert batch["phase_y"].shape[0] == BATCH_SIZE


def test_train_val_splits_are_non_empty_and_different():
    train_ds = build_unpaired_cqt_dataset(
        cache_root=CACHE_ROOT,
        snippet_seconds=SNIPPET_SECONDS,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        windows_per_track=1,
        return_chroma=False,
        return_phase=False,
        return_metadata=False,
        split="train",
        val_fraction=0.1,
        split_seed=1337,
    )

    val_ds = build_unpaired_cqt_dataset(
        cache_root=CACHE_ROOT,
        snippet_seconds=SNIPPET_SECONDS,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        windows_per_track=1,
        return_chroma=False,
        return_phase=False,
        return_metadata=False,
        split="val",
        val_fraction=0.1,
        split_seed=1337,
    )

    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(train_ds) != len(val_ds)

