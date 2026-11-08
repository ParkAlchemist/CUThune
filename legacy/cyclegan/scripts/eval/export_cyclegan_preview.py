from __future__ import annotations

from pathlib import Path
import runpy
import argparse
import json
import csv
from typing import Any

from legacy.cyclegan.losses.cyclegan_losses import cqt_to_chroma
from legacy.cyclegan.models.cyclegan import AudioResnetGenerator, PatchGANDiscriminator, GeneratorConfig, \
    DiscriminatorConfig

PROJECT_ROOT = Path(__file__).resolve().parents[4]

runpy.run_path(
    str(PROJECT_ROOT / "scripts" / "eval" / "export_cyclegan_preview.py"),
    run_name="__main__",
)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

from shared.src.chiptune_shared.cqt.cache import CachedCQTTrackDataset
from cut.src.nes_cut.data.unpaired_dataset import UnpairedCQTDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fixed CycleGAN CQT preview grids from a checkpoint."
    )

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)

    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=1337)

    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--snippet-seconds", type=float, default=4.0)
    parser.add_argument("--windows-per-track", type=int, default=1)
    parser.add_argument("--min-window-energy", type=float, default=0.01)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--save-tensors", action="store_true")

    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(path, map_location=device)


def build_models_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[
    AudioResnetGenerator,
    AudioResnetGenerator,
    PatchGANDiscriminator | None,
    PatchGANDiscriminator | None,
]:
    gen_cfg_dict = checkpoint.get("generator_config", {})
    disc_cfg_dict = checkpoint.get("discriminator_config", {})

    gen_cfg = GeneratorConfig(**gen_cfg_dict)
    disc_cfg = DiscriminatorConfig(**disc_cfg_dict) if disc_cfg_dict else None

    g_x_to_y = AudioResnetGenerator(gen_cfg).to(device)
    g_y_to_x = AudioResnetGenerator(gen_cfg).to(device)

    g_x_to_y.load_state_dict(checkpoint["g_x_to_y"])
    g_y_to_x.load_state_dict(checkpoint["g_y_to_x"])

    g_x_to_y.eval()
    g_y_to_x.eval()

    d_x = None
    d_y = None

    if disc_cfg is not None and "d_x" in checkpoint and "d_y" in checkpoint:
        d_x = PatchGANDiscriminator(disc_cfg).to(device)
        d_y = PatchGANDiscriminator(disc_cfg).to(device)

        d_x.load_state_dict(checkpoint["d_x"])
        d_y.load_state_dict(checkpoint["d_y"])

        d_x.eval()
        d_y.eval()

    return g_x_to_y, g_y_to_x, d_x, d_y


def build_fixed_preview_dataset(args: argparse.Namespace) -> UnpairedCQTDataset:
    poly_ds = CachedCQTTrackDataset(
        cache_root=args.cache_root / "poly",
        domain="poly",
        snippet_seconds=args.snippet_seconds,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        windows_per_track=args.windows_per_track,
        random_window=False,
        random_track=False,
        return_chroma=True,
        return_phase=False,
        return_metadata=True,
        min_window_energy=args.min_window_energy,
        track_cache_size=4,
        split=args.split,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )

    chip_ds = CachedCQTTrackDataset(
        cache_root=args.cache_root / "chip",
        domain="chip",
        snippet_seconds=args.snippet_seconds,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        windows_per_track=args.windows_per_track,
        random_window=False,
        random_track=False,
        return_chroma=True,
        return_phase=False,
        return_metadata=True,
        min_window_energy=args.min_window_energy,
        track_cache_size=4,
        split=args.split,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )

    return UnpairedCQTDataset(
        poly_dataset=poly_ds,
        chip_dataset=chip_ds,
        length_mode="min",
        random_y=False,
    )


def tensor_to_cqt_image(x: torch.Tensor) -> torch.Tensor:
    """
    Accepts:
        [1, F, T], [F, T], or [1, 1, F, T]

    Returns:
        [F, T] CPU float tensor.
    """
    x = x.detach().cpu().float()

    if x.ndim == 4:
        x = x[0, 0]
    elif x.ndim == 3:
        if x.shape[0] == 1:
            x = x[0]
    elif x.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected CQT tensor shape: {tuple(x.shape)}")

    return x


def plot_cqt(ax, cqt: torch.Tensor, title: str) -> None:
    image = tensor_to_cqt_image(cqt).numpy()

    ax.imshow(
        image,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=-1.0,
        vmax=1.0,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Time frames", fontsize=8)
    ax.set_ylabel("CQT bins", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)


def l1_value(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.l1_loss(a.detach().float(), b.detach().float()).cpu().item())


def chroma_l1_value(a: torch.Tensor, b: torch.Tensor) -> float:
    ca = cqt_to_chroma(a.detach().float())
    cb = cqt_to_chroma(b.detach().float())
    return float(F.l1_loss(ca, cb).cpu().item())


def tensor_stats(prefix: str, x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float()
    return {
        f"{prefix}_min": float(x.min().cpu().item()),
        f"{prefix}_max": float(x.max().cpu().item()),
        f"{prefix}_mean": float(x.mean().cpu().item()),
        f"{prefix}_std": float(x.std().cpu().item()),
    }


def save_preview_grid(
    out_path: Path,
    real_x: torch.Tensor,
    fake_y: torch.Tensor,
    rec_x: torch.Tensor,
    real_y: torch.Tensor,
    id_y: torch.Tensor,
    fake_x: torch.Tensor,
    sample_index: int,
    dpi: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 7),
        constrained_layout=True,
    )

    plot_cqt(axes[0, 0], real_x, "Input poly X")
    plot_cqt(axes[0, 1], fake_y, "Generated chip G_XY(X)")
    plot_cqt(axes[0, 2], rec_x, "Cycle reconstruction G_YX(G_XY(X))")

    plot_cqt(axes[1, 0], real_y, "Real chip Y")
    plot_cqt(axes[1, 1], id_y, "Identity chip G_XY(Y)")
    plot_cqt(axes[1, 2], fake_x, "Generated poly G_YX(Y)")

    fig.suptitle(f"CycleGAN CQT preview sample {sample_index}", fontsize=13)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_metric_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, device=device)
    global_step = checkpoint.get("global_step", None)
    epoch = checkpoint.get("epoch", None)

    g_x_to_y, g_y_to_x, d_x, d_y = build_models_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
    )

    dataset = build_fixed_preview_dataset(args)

    num_samples = min(args.num_samples, len(dataset))
    metrics: list[dict[str, Any]] = []

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Epoch: {epoch}")
    print(f"Global step: {global_step}")
    print(f"Preview samples: {num_samples}")
    print(f"Output dir: {args.out_dir}")

    for sample_idx in tqdm(range(num_samples), desc="Exporting previews"):
        item = dataset[sample_idx]

        real_x = item["real_x"].unsqueeze(0).to(device)
        real_y = item["real_y"].unsqueeze(0).to(device)

        fake_y = torch.clamp(g_x_to_y(real_x), -1.0, 1.0)
        fake_x = torch.clamp(g_y_to_x(real_y), -1.0, 1.0)

        rec_x = torch.clamp(g_y_to_x(fake_y), -1.0, 1.0)
        rec_y = torch.clamp(g_x_to_y(fake_x), -1.0, 1.0)

        id_x = torch.clamp(g_y_to_x(real_x), -1.0, 1.0)
        id_y = torch.clamp(g_x_to_y(real_y), -1.0, 1.0)

        preview_path = args.out_dir / f"preview_{sample_idx:04d}.png"

        save_preview_grid(
            out_path=preview_path,
            real_x=real_x,
            fake_y=fake_y,
            rec_x=rec_x,
            real_y=real_y,
            id_y=id_y,
            fake_x=fake_x,
            sample_index=sample_idx,
            dpi=args.dpi,
        )

        row: dict[str, Any] = {
            "sample_index": sample_idx,
            "epoch": epoch,
            "global_step": global_step,
            "preview_path": str(preview_path),

            "source_path_poly": item.get("source_path_poly", ""),
            "source_path_chip": item.get("source_path_chip", ""),

            "frame_start_poly": int(item.get("frame_start_poly", -1)),
            "frame_end_poly": int(item.get("frame_end_poly", -1)),
            "frame_start_chip": int(item.get("frame_start_chip", -1)),
            "frame_end_chip": int(item.get("frame_end_chip", -1)),

            "cycle_x_l1": l1_value(real_x, rec_x),
            "cycle_y_l1": l1_value(real_y, rec_y),
            "identity_x_l1": l1_value(real_x, id_x),
            "identity_y_l1": l1_value(real_y, id_y),
            "chroma_x_fake_y_l1": chroma_l1_value(real_x, fake_y),
            "chroma_y_fake_x_l1": chroma_l1_value(real_y, fake_x),
        }

        row.update(tensor_stats("real_x", real_x))
        row.update(tensor_stats("fake_y", fake_y))
        row.update(tensor_stats("real_y", real_y))
        row.update(tensor_stats("id_y", id_y))

        if d_y is not None:
            pred_real_y = d_y(real_y)
            pred_fake_y = d_y(fake_y)
            row["d_y_real_mean"] = float(pred_real_y.mean().cpu().item())
            row["d_y_fake_mean"] = float(pred_fake_y.mean().cpu().item())

        if d_x is not None:
            pred_real_x = d_x(real_x)
            pred_fake_x = d_x(fake_x)
            row["d_x_real_mean"] = float(pred_real_x.mean().cpu().item())
            row["d_x_fake_mean"] = float(pred_fake_x.mean().cpu().item())

        metrics.append(row)

        if args.save_tensors:
            tensor_dir = args.out_dir / "tensors"
            tensor_dir.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "real_x": real_x.detach().cpu(),
                    "fake_y": fake_y.detach().cpu(),
                    "rec_x": rec_x.detach().cpu(),
                    "real_y": real_y.detach().cpu(),
                    "id_y": id_y.detach().cpu(),
                    "fake_x": fake_x.detach().cpu(),
                    "rec_y": rec_y.detach().cpu(),
                    "id_x": id_x.detach().cpu(),
                    "metadata": {
                        "sample_index": sample_idx,
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                },
                tensor_dir / f"preview_{sample_idx:04d}.pt",
            )

    save_metric_csv(args.out_dir / "metrics.csv", metrics)

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "epoch": epoch,
                "global_step": global_step,
                "num_samples": num_samples,
                "cache_root": str(args.cache_root),
                "split": args.split,
                "val_fraction": args.val_fraction,
                "split_seed": args.split_seed,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Preview export complete.")
    print(f"Wrote: {args.out_dir}")
    print(f"Metrics: {args.out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()

