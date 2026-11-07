from __future__ import annotations

from pathlib import Path
import sys
import argparse
import csv
from typing import Any

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from src.models.cyclegan import (
    GeneratorConfig,
    DiscriminatorConfig,
    AudioResnetGenerator,
    PatchGANDiscriminator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def load_models(checkpoint: dict[str, Any], device: torch.device):
    gen_cfg = GeneratorConfig(**checkpoint["generator_config"])
    disc_cfg = DiscriminatorConfig(**checkpoint["discriminator_config"])

    g_x_to_y = AudioResnetGenerator(gen_cfg).to(device)
    g_y_to_x = AudioResnetGenerator(gen_cfg).to(device)
    d_x = PatchGANDiscriminator(disc_cfg).to(device)
    d_y = PatchGANDiscriminator(disc_cfg).to(device)

    g_x_to_y.load_state_dict(checkpoint["g_x_to_y"])
    g_y_to_x.load_state_dict(checkpoint["g_y_to_x"])
    d_x.load_state_dict(checkpoint["d_x"])
    d_y.load_state_dict(checkpoint["d_y"])

    return {
        "g_x_to_y": g_x_to_y,
        "g_y_to_x": g_y_to_x,
        "d_x": d_x,
        "d_y": d_y,
    }


def collect_param_stats(model_name: str, model: torch.nn.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for name, param in model.named_parameters():
        data = param.detach().cpu().float()

        rows.append({
            "model": model_name,
            "parameter": name,
            "shape": "x".join(str(x) for x in data.shape),
            "numel": int(data.numel()),
            "mean": float(data.mean().item()),
            "std": float(data.std().item()) if data.numel() > 1 else 0.0,
            "min": float(data.min().item()),
            "max": float(data.max().item()),
            "l1_norm": float(data.abs().sum().item()),
            "l2_norm": float(torch.linalg.vector_norm(data).item()),
            "max_abs": float(data.abs().max().item()),
        })

    return rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_weight_histograms(path: Path, model_name: str, model: torch.nn.Module) -> None:
    weights = []

    for name, param in model.named_parameters():
        if param.ndim >= 2:
            weights.append(param.detach().cpu().float().flatten())

    if not weights:
        return

    all_weights = torch.cat(weights).numpy()

    plt.figure(figsize=(10, 5))
    plt.hist(all_weights, bins=200)
    plt.title(f"{model_name} weight histogram")
    plt.xlabel("Weight value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_norm_bars(path: Path, rows: list[dict[str, Any]], model_name: str, top_k: int) -> None:
    model_rows = [
        row for row in rows
        if row["model"] == model_name and ".weight" in row["parameter"]
    ]

    model_rows = sorted(model_rows, key=lambda r: r["l2_norm"], reverse=True)[:top_k]

    if not model_rows:
        return

    labels = [row["parameter"] for row in model_rows]
    values = [row["l2_norm"] for row in model_rows]

    plt.figure(figsize=(12, max(6, len(labels) * 0.25)))
    plt.barh(range(len(labels)), values)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.xlabel("L2 norm")
    plt.title(f"{model_name} top-{top_k} parameter L2 norms")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def find_first_conv_weight(model: torch.nn.Module) -> tuple[str, torch.Tensor] | None:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            return name, module.weight.detach().cpu().float()
    return None


def plot_first_conv_filters(path: Path, model_name: str, model: torch.nn.Module) -> None:
    result = find_first_conv_weight(model)
    if result is None:
        return

    layer_name, weight = result

    # Shape: [out_channels, in_channels, kh, kw]
    if weight.ndim != 4:
        return

    out_channels = min(weight.shape[0], 32)
    filters = weight[:out_channels, 0]

    cols = 8
    rows = (out_channels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 2 * rows))
    axes = axes.flatten()

    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")

        if i < out_channels:
            image = filters[i].numpy()
            vmax = max(abs(image.min()), abs(image.max()))
            ax.imshow(image, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{i}", fontsize=8)

    fig.suptitle(f"{model_name} first Conv2d filters: {layer_name}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_attention_values(path: Path, model_name: str, model: torch.nn.Module) -> list[dict[str, Any]]:
    rows = []

    for name, param in model.named_parameters():
        if name.endswith("gamma"):
            rows.append({
                "model": model_name,
                "parameter": name,
                "value": float(param.detach().cpu().item()),
            })

    if rows:
        save_csv(path, rows)

    return rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    models = load_models(checkpoint, device)

    all_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []

    for model_name, model in models.items():
        model.eval()

        stats = collect_param_stats(model_name, model)
        all_rows.extend(stats)

        plot_weight_histograms(
            args.out_dir / f"{model_name}_weight_histogram.png",
            model_name,
            model,
        )

        plot_norm_bars(
            args.out_dir / f"{model_name}_weight_norms.png",
            stats,
            model_name,
            top_k=args.top_k,
        )

        plot_first_conv_filters(
            args.out_dir / f"{model_name}_first_conv_filters.png",
            model_name,
            model,
        )

        attention_rows.extend(
            write_attention_values(
                args.out_dir / f"{model_name}_attention_gamma.csv",
                model_name,
                model,
            )
        )

    save_csv(args.out_dir / "parameter_stats.csv", all_rows)
    save_csv(args.out_dir / "attention_gamma_all.csv", attention_rows)

    print("Weight visualization complete.")
    print(f"Wrote: {args.out_dir}")


if __name__ == "__main__":
    main()

