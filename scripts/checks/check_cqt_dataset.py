from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cut.src.nes_cut.data.unpaired_dataset import build_unpaired_cqt_dataloader


def main() -> None:
    cache_root = Path("E:/Projects/Datasets/cache/cqt")

    loader = build_unpaired_cqt_dataloader(
        cache_root=cache_root,
        batch_size=2,
        num_workers=0,
        snippet_seconds=4.0,
        sample_rate=22050,
        hop_length=512,
        windows_per_track=4,
        return_chroma=True,
        return_phase=True,
        return_metadata=False,
        min_window_energy=0.01,
    )

    batch = next(iter(loader))

    print("Batch keys:")
    for key in sorted(batch.keys()):
        value = batch[key]
        if hasattr(value, "shape"):
            print(f"  {key}: {tuple(value.shape)} {value.dtype}")
        else:
            print(f"  {key}: {type(value)}")

    print("\nImportant shapes:")
    print("  real_x:", batch["real_x"].shape)
    print("  real_y:", batch["real_y"].shape)

    if "chroma_x" in batch:
        print("  chroma_x:", batch["chroma_x"].shape)

    if "chroma_y" in batch:
        print("  chroma_y:", batch["chroma_y"].shape)

    if "phase_x" in batch:
        print("  phase_x:", batch["phase_x"].shape)

    if "phase_y" in batch:
        print("  phase_y:", batch["phase_y"].shape)

    print("\nEnergy:")
    print("  poly:", batch["energy_poly"])
    print("  chip:", batch["energy_chip"])

    assert batch["real_x"].ndim == 4
    assert batch["real_y"].ndim == 4

    assert batch["real_x"].shape[1] == 1
    assert batch["real_y"].shape[1] == 1

    assert batch["real_x"].shape[2] == 96
    assert batch["real_y"].shape[2] == 96

    print("\nCQT dataset smoke test passed.")


if __name__ == "__main__":
    main()
