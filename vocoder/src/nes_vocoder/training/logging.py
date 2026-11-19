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
import torch.nn.functional as F

from shared.src.chiptune_shared.nn.blocks.convnext import ConvNeXtContextBlock1d, MultiKernelConvNeXtContextBlock1d
from shared.src.chiptune_shared.nn.blocks.global_response_norm import GlobalResponseNorm1d
from shared.src.chiptune_shared.nn.blocks.temporal_attention import RoPETemporalSelfAttention1d
from vocoder.src.nes_vocoder.config.schema import VocoderExperimentConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vocoder.src.nes_vocoder.training.context import (
        RunPaths,
        TrainingState,
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


def waveform_diagnostic_metrics(
    *,
    pre_tanh: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    silence_window_size: int = 256,
    silence_rms_threshold: float = 1e-3,
    eps: float = 1e-8,
) -> dict[str, float]:
    pre = pre_tanh.detach().float()
    pred = prediction.detach().float()
    real = target.detach().float()

    pre_abs = pre.abs()
    pred_abs = pred.abs()

    tanh_derivative = (
        1.0 - pred.square()
    ).clamp_min(0.0)

    pred_rms = torch.sqrt(
        pred.square().mean() + eps
    )

    target_rms = torch.sqrt(
        real.square().mean() + eps
    )

    pred_peak = pred_abs.amax()
    target_peak = real.abs().amax()

    target_local_rms = torch.sqrt(
        F.avg_pool1d(
            real.square(),
            kernel_size=silence_window_size,
            stride=1,
            padding=silence_window_size // 2,
            count_include_pad=False,
        )[..., :real.shape[-1]]
        + eps
    )

    silence_mask = (
        target_local_rms
        <= silence_rms_threshold
    )

    silence_fraction = (
        silence_mask.float().mean()
    )

    if bool(silence_mask.any()):
        silent_pre = pre[silence_mask]
        silent_pred = pred[silence_mask]

        silence_pre_mean = (
            silent_pre.mean()
        )
        silence_pre_min = (
            silent_pre.amin()
        )
        silence_pre_max = (
            silent_pre.amax()
        )
        silence_output_mean = (
            silent_pred.mean()
        )
        silence_output_rms = torch.sqrt(
            silent_pred.square().mean()
            + eps
        )
        silence_negative_rail = (
            silent_pred <= -0.99
        ).float().mean()
        silence_positive_rail = (
            silent_pred >= 0.99
        ).float().mean()
    else:
        zero = pred.new_zeros(())

        silence_pre_mean = zero
        silence_pre_min = zero
        silence_pre_max = zero
        silence_output_mean = zero
        silence_output_rms = zero
        silence_negative_rail = zero
        silence_positive_rail = zero

    return {
        "generator_output/rms": float(pred_rms.cpu()),
        "generator_output/target_rms": float(target_rms.cpu()),
        "generator_output/rms_ratio": float((pred_rms / target_rms.clamp_min(eps)).cpu()),
        "generator_output/peak": float(pred_peak.cpu()),
        "generator_output/target_peak": float(target_peak.cpu()),
        "generator_output/peak_ratio": float((pred_peak / target_peak.clamp_min(eps)).cpu()),
        "generator_output/mean": float(pred.mean().cpu()),
        "generator_output/target_mean": float(real.mean().cpu()),
        "generator_output/std": float(pred.std(unbiased=False).cpu()),

        "tanh/pre_abs_mean": float(pre_abs.mean().cpu()),
        "tanh/pre_abs_p95": float(torch.quantile(pre_abs.flatten(),0.95,).cpu()),
        "tanh/pre_abs_p99": float(torch.quantile(pre_abs.flatten(),0.99,).cpu()),
        "tanh/pre_abs_max": float(pre_abs.amax().cpu()),
        "tanh/pre_mean": float(pre.mean().cpu()),

        "tanh/output_ge_090": float((pred_abs >= 0.90).float().mean().cpu()),
        "tanh/output_ge_095": float((pred_abs >= 0.95).float().mean().cpu()),
        "tanh/output_ge_099": float((pred_abs >= 0.99).float().mean().cpu()),
        "tanh/output_ge_0999": float((pred_abs >= 0.999).float().mean().cpu()),
        "tanh/positive_ge_099": float((pred >= 0.99).float().mean().cpu()),
        "tanh/negative_le_m099": float((pred <= -0.99).float().mean().cpu()),
        "tanh/derivative_mean": float(tanh_derivative.mean().cpu()),
        "tanh/derivative_lt_01": float((tanh_derivative < 0.1).float().mean().cpu()),
        "tanh/derivative_lt_001": float((tanh_derivative < 0.01).float().mean().cpu()),

        "silence/target_fraction": float(silence_fraction.cpu()),
        "silence/pre_tanh_mean": float(silence_pre_mean.cpu()),
        "silence/pre_tanh_min": float(silence_pre_min.cpu()),
        "silence/pre_tanh_max": float(silence_pre_max.cpu()),
        "silence/output_mean": float(silence_output_mean.cpu()),
        "silence/output_rms": float(silence_output_rms.cpu()),
        "silence/output_negative_rail": float(silence_negative_rail.cpu()),
        "silence/output_positive_rail": float(silence_positive_rail.cpu()),
    }


def collect_generator_output_parameter_metrics(
    generator: nn.Module,
) -> dict[str, float]:
    conv_post = getattr(
        generator,
        "conv_post",
        None,
    )

    if conv_post is None:
        return {}

    weight = (
        conv_post.weight
        .detach()
        .float()
    )

    metrics = {
        "generator_output_layer/"
        "weight_norm": float(
            weight.norm().cpu()
        ),
        "generator_output_layer/"
        "weight_mean_abs": float(
            weight.abs().mean().cpu()
        ),
        "generator_output_layer/"
        "weight_max_abs": float(
            weight.abs().max().cpu()
        ),
    }

    bias = getattr(
        conv_post,
        "bias",
        None,
    )

    if bias is not None:
        bias_detached = (
            bias.detach().float()
        )

        metrics.update({
            "generator_output_layer/"
            "bias_mean": float(
                bias_detached.mean().cpu()
            ),
            "generator_output_layer/"
            "bias_min": float(
                bias_detached.min().cpu()
            ),
            "generator_output_layer/"
            "bias_max": float(
                bias_detached.max().cpu()
            ),
            "generator_output_layer/"
            "bias_trainable": float(
                bias.requires_grad
            ),
        })

    return metrics


