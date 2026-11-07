from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn as nn
from torch.nn import functional as F

from vocoder.src.nes_vocoder.config.schema import DCOffsetLossConfig


@dataclass(frozen=True)
class DCOffsetScaleLossOutput:
    window_size: int
    hop_size: int
    loss: torch.Tensor


@dataclass(frozen=True)
class MultiScaleDCOffsetLossOutput:
    total: torch.Tensor

    local: torch.Tensor
    global_mean: torch.Tensor

    scales: tuple[
        DCOffsetScaleLossOutput,
        ...,
    ]


class MultiScaleDCOffsetLoss(nn.Module):
    """
    Match target-relative local and global waveform means.

    This loss is phase-relaxed. It does not require exact
    sample alignment and does not force every target segment
    to have zero mean.
    """

    def __init__(
        self,
        config: DCOffsetLossConfig,
    ) -> None:
        super().__init__()

        if not config.window_sizes:
            raise ValueError(
                "DC-offset loss requires at least "
                "one window size."
            )

        if any(
            window_size <= 0
            for window_size
            in config.window_sizes
        ):
            raise ValueError(
                "All DC-offset window sizes "
                "must be positive."
            )

        if len(
            set(config.window_sizes)
        ) != len(config.window_sizes):
            raise ValueError(
                "DC-offset window sizes "
                "must be unique."
            )

        if config.hop_ratio <= 0:
            raise ValueError(
                "DC-offset hop_ratio "
                "must be positive."
            )

        self.window_sizes = tuple(
            sorted(config.window_sizes)
        )

        self.hop_ratio = config.hop_ratio

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> MultiScaleDCOffsetLossOutput:
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target waveform "
                "shapes differ: "
                f"{tuple(prediction.shape)} versus "
                f"{tuple(target.shape)}."
            )

        if prediction.ndim != 3:
            raise ValueError(
                "Expected waveform [B,C,T], "
                f"got {tuple(prediction.shape)}."
            )

        scale_outputs: list[
            DCOffsetScaleLossOutput
        ] = []

        for window_size in self.window_sizes:
            if (
                prediction.shape[-1]
                < window_size
            ):
                raise ValueError(
                    "Waveform is shorter than a "
                    "configured DC-offset window. "
                    f"Length={prediction.shape[-1]}, "
                    f"window_size={window_size}."
                )

            hop_size = max(
                1,
                window_size // self.hop_ratio,
            )

            prediction_mean = F.avg_pool1d(
                prediction,
                kernel_size=window_size,
                stride=hop_size,
                padding=0,
                count_include_pad=False,
            )

            target_mean = F.avg_pool1d(
                target,
                kernel_size=window_size,
                stride=hop_size,
                padding=0,
                count_include_pad=False,
            )

            scale_loss = F.l1_loss(
                prediction_mean,
                target_mean,
                reduction="mean",
            )

            scale_outputs.append(
                DCOffsetScaleLossOutput(
                    window_size=window_size,
                    hop_size=hop_size,
                    loss=scale_loss,
                )
            )

        local_loss = torch.stack(
            [
                output.loss
                for output in scale_outputs
            ]
        ).mean()

        prediction_global_mean = (
            prediction.mean(dim=-1)
        )

        target_global_mean = (
            target.mean(dim=-1)
        )

        global_loss = F.l1_loss(
            prediction_global_mean,
            target_global_mean,
            reduction="mean",
        )

        # Give the local and whole-segment objectives
        # equal aggregate authority.
        total = 0.5 * (
            local_loss + global_loss
        )

        return MultiScaleDCOffsetLossOutput(
            total=total,
            local=local_loss,
            global_mean=global_loss,
            scales=tuple(scale_outputs),
        )
