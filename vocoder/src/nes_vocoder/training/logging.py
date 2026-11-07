from __future__ import annotations

from dataclasses import asdict
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import json
import shutil
import time

import torch
import torch.nn as nn

from vocoder.src.nes_vocoder.config.schema import VocoderExperimentConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.vocoder import (
        RunPaths,
        TrainingState,
    )
from src.models.blocks import (
    GlobalResponseNorm1d,
    MultiKernelConvNeXtContextBlock1d,
    ConvNeXtContextBlock1d,
)
from src.models.blocks import (
    RoPETemporalSelfAttention1d,
)


def save_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(path)


def save_config_snapshot(
    *,
    config: VocoderExperimentConfig,
    source_path: Path,
    paths: RunPaths,
) -> None:
    save_json(
        paths.resolved_config_path,
        asdict(config),
    )

    shutil.copy2(
        source_path,
        paths.source_config_path,
    )


def write_status(
    *,
    paths: RunPaths,
    state: TrainingState,
    losses: dict[str, float],
) -> None:
    save_json(
        paths.status_path,
        {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "micro_step": state.micro_step,
            "segments_seen": state.segments_seen,
            "audio_samples_seen": (
                state.audio_samples_seen
            ),
            "time": time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "losses": losses,
            "latest_checkpoint": (
                state.latest_checkpoint
            ),
            "latest_preview": state.latest_preview,
        },
    )


def summarize_output_tensors(
    outputs: Sequence[torch.Tensor],
) -> dict[str, float]:
    if not outputs:
        return {}

    values = torch.cat(
        [
            output.detach().float().reshape(-1)
            for output in outputs
        ]
    )

    sigmoid_values = torch.sigmoid(values)

    return {
        "score_mean": float(values.mean().cpu()),
        "score_std": float(
            values.std(unbiased=False).cpu()
        ),
        "score_min": float(values.min().cpu()),
        "score_max": float(values.max().cpu()),
        "sigmoid_mean": float(
            sigmoid_values.mean().cpu()
        ),
        "sigmoid_std": float(
            sigmoid_values.std(
                unbiased=False
            ).cpu()
        ),
    }


def discriminator_health_metrics(
    outputs: dict[str, list],
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    for family in ("combined", "mpd", "msd", "mrd"):
        if family == "combined":
            real_outputs = outputs["real_outputs"]
            fake_outputs = outputs["fake_outputs"]
        else:
            real_outputs = outputs[
                f"{family}_real_outputs"
            ]
            fake_outputs = outputs[
                f"{family}_fake_outputs"
            ]

        for domain, tensors in (
            ("real", real_outputs),
            ("fake", fake_outputs),
        ):
            statistics = summarize_output_tensors(
                tensors
            )

            for statistic, value in statistics.items():
                metrics[
                    "discriminator_health/"
                    f"{family}/{domain}_{statistic}"
                ] = value

    return metrics


def discriminator_branch_metrics(
    outputs: dict[str, list],
    *,
    mpd_periods: tuple[int, ...],
    mrd_resolutions: tuple[
        tuple[int, int, int],
        ...,
    ],
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    mpd_real_outputs = outputs["mpd_real_outputs"]
    mpd_fake_outputs = outputs["mpd_fake_outputs"]

    if len(mpd_periods) != len(mpd_real_outputs):
        raise RuntimeError(
            "MPD branch metadata does not match realized " 
            "discriminator outputs: " 
            f"periods={len(mpd_periods)}, " 
            f"real_outputs={len(mpd_real_outputs)}."
        )

    if len(mpd_real_outputs) != len(mpd_fake_outputs):
        raise RuntimeError("MPD real and fake branch counts differ.")

    mrd_real_outputs = outputs["mrd_real_outputs"]
    mrd_fake_outputs = outputs["mrd_fake_outputs"]

    if len(mrd_resolutions) != len(mrd_real_outputs):
        raise RuntimeError(
            "MRD branch metadata does not match realized " 
            "discriminator outputs: " 
            f"resolutions={len(mrd_resolutions)}, " 
            f"real_outputs={len(mrd_real_outputs)}."
        )

    if len(mrd_real_outputs) != len(mrd_fake_outputs):
        raise RuntimeError("MRD real and fake branch counts differ.")

    branch_groups = (
        (
            "mpd",
            [
                f"period_{period}"
                for period in mpd_periods
            ],
        ),
        (
            "msd",
            [
                f"scale_{index}"
                for index in range(
                    len(outputs["msd_real_outputs"])
                )
            ],
        ),
        (
            "mrd",
            [
                f"fft_{fft}_hop_{hop}"
                for fft, hop, _ in mrd_resolutions
            ],
        ),
    )

    for family, branch_names in branch_groups:
        real_outputs = outputs[
            f"{family}_real_outputs"
        ]
        fake_outputs = outputs[
            f"{family}_fake_outputs"
        ]

        for branch_name, real, fake in zip(
            branch_names,
            real_outputs,
            fake_outputs,
        ):
            real_values = (
                real.detach().float().reshape(-1)
            )
            fake_values = (
                fake.detach().float().reshape(-1)
            )

            prefix = (
                "discriminator_health/"
                f"{family}/{branch_name}"
            )

            metrics[
                f"{prefix}/real_score_mean"
            ] = float(real_values.mean().cpu())

            metrics[
                f"{prefix}/fake_score_mean"
            ] = float(fake_values.mean().cpu())

            metrics[
                f"{prefix}/real_sigmoid_mean"
            ] = float(
                torch.sigmoid(real_values).mean().cpu()
            )

            metrics[
                f"{prefix}/fake_sigmoid_mean"
            ] = float(
                torch.sigmoid(fake_values).mean().cpu()
            )

    return metrics


def collect_context_parameter_metrics(
    generator: nn.Module,
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    convnext_index = 0
    attention_index = 0
    grn_index = 0

    for module in generator.modules():
        if isinstance(
            module,
            (
                ConvNeXtContextBlock1d,
                MultiKernelConvNeXtContextBlock1d,
            ),
        ):
            scale = (
                module.layer_scale
                .detach()
                .float()
            )

            prefix = (
                f"context/convnext_{convnext_index:02d}"
            )

            metrics[
                f"{prefix}/layer_scale_mean"
            ] = float(scale.mean().cpu())

            metrics[
                f"{prefix}/layer_scale_mean_abs"
            ] = float(
                scale.abs().mean().cpu()
            )

            metrics[
                f"{prefix}/layer_scale_max_abs"
            ] = float(
                scale.abs().max().cpu()
            )

            convnext_index += 1

        elif isinstance(
            module,
            RoPETemporalSelfAttention1d,
        ):
            scale = (
                module.layer_scale
                .detach()
                .float()
            )

            prefix = (
                f"context/attention_{attention_index:02d}"
            )

            metrics[
                f"{prefix}/layer_scale_mean"
            ] = float(scale.mean().cpu())

            metrics[
                f"{prefix}/layer_scale_mean_abs"
            ] = float(
                scale.abs().mean().cpu()
            )

            metrics[
                f"{prefix}/layer_scale_max_abs"
            ] = float(
                scale.abs().max().cpu()
            )

            attention_index += 1

        elif isinstance(
            module,
            GlobalResponseNorm1d,
        ):
            gamma = module.gamma.detach().float()
            beta = module.beta.detach().float()

            prefix = f"context/grn_{grn_index:02d}"

            metrics[
                f"{prefix}/gamma_mean_abs"
            ] = float(
                gamma.abs().mean().cpu()
            )

            metrics[
                f"{prefix}/gamma_max_abs"
            ] = float(
                gamma.abs().max().cpu()
            )

            metrics[
                f"{prefix}/beta_mean_abs"
            ] = float(
                beta.abs().mean().cpu()
            )

            grn_index += 1

    return metrics


def write_model_structure(
    *,
    generator: nn.Module,
    discriminator: nn.Module,
    output_path: Path,
) -> None:
    generator_parameters = sum(
        parameter.numel()
        for parameter in generator.parameters()
    )

    discriminator_parameters = sum(
        parameter.numel()
        for parameter in discriminator.parameters()
    )

    text = (
        "GENERATOR\n"
        "=========\n"
        f"Parameters: {generator_parameters:,}\n\n"
        f"{generator}\n\n"
        "DISCRIMINATOR\n"
        "=============\n"
        f"Parameters: {discriminator_parameters:,}\n\n"
        f"{discriminator}\n"
    )

    output_path.write_text(
        text,
        encoding="utf-8",
    )


def module_type_counts(
    module: nn.Module,
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                type(child).__name__
                for child in module.modules()
            ).items()
        )
    )


def write_model_manifest(
    *,
    generator: nn.Module,
    discriminator: nn.Module,
    output_path: Path,
) -> dict:
    manifest = {
        "generator": {
            "parameter_count": sum(
                parameter.numel()
                for parameter
                in generator.parameters()
            ),
            "module_type_counts": (
                module_type_counts(generator)
            ),
        },
        "discriminator": {
            "parameter_count": sum(
                parameter.numel()
                for parameter
                in discriminator.parameters()
            ),
            "module_type_counts": (
                module_type_counts(discriminator)
            ),
        },
    }

    save_json(output_path, manifest)

    return manifest


def write_torchinfo_generator_summary(
    *,
    generator: nn.Module,
    output_path: Path,
    cqt_bins: int,
    segment_frames: int,
    device: torch.device,
) -> str:
    from torchinfo import summary

    was_training = generator.training
    generator.eval()

    try:
        result = summary(
            generator,
            input_size=(
                1,
                cqt_bins,
                segment_frames,
            ),
            device=str(device),
            depth=7,
            col_names=(
                "input_size",
                "output_size",
                "num_params",
                "trainable",
                "mult_adds",
            ),
            verbose=0,
        )

        text = str(result)

        output_path.write_text(
            text,
            encoding="utf-8",
        )

        return text
    finally:
        generator.train(was_training)


