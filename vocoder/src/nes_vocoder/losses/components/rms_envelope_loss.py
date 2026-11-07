from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn as nn
from torch.nn import functional as F

from vocoder.src.nes_vocoder.config.schema import RMSEnvelopeLossConfig


@dataclass(frozen=True)
class RMSEnvelopeScaleLossOutput:
    window_size: int
    hop_size: int
    loss: torch.Tensor


@dataclass(frozen=True)
class MultiScaleRMSEnvelopeLossOutput:
    total: torch.Tensor

    scales: tuple[
        RMSEnvelopeScaleLossOutput,
        ...,
    ]


def rms_envelope(
        waveform: torch.Tensor,
        *,
        window_size: int,
        hop_size: int,
        eps: float,
) -> torch.Tensor:
    if waveform.ndim != 3:
        raise ValueError(
            f"Waveform shape must be three dimensional, got {tuple(waveform.shape)}"
        )

    if window_size <= 0:
        raise ValueError(
            f"Window size must be positive, got {window_size}"
        )

    if hop_size <= 0:
        raise ValueError(
            f"Hop size must be positive, got {hop_size}"
        )

    if eps <= 0:
        raise ValueError(
            f"Eps must be positive, got {eps}"
        )

    if waveform.shape[-1] < window_size:
        raise ValueError(
            f"Waveform must be at least as long as the RMS window. "
            f"Length={waveform.shape[-1]}, window_size={window_size}."
        )

    power = waveform.square()

    mean_power = F.avg_pool1d(
        power,
        kernel_size=window_size,
        stride=hop_size,
        padding=0,
        count_include_pad=False,
    )

    return torch.sqrt(mean_power + eps)


class MultiScaleRMSEnvelopeLoss(nn.Module):
    def __init__(
        self,
        config: RMSEnvelopeLossConfig,
    ) -> None:
        super().__init__()

        if not config.window_sizes:
            raise ValueError(
                "RMS-envelope loss requires at least "
                "one window size."
            )

        if any(
            window_size <= 0
            for window_size
            in config.window_sizes
        ):
            raise ValueError(
                "All RMS-envelope window sizes "
                "must be positive."
            )

        if len(set(config.window_sizes)) != len(
            config.window_sizes
        ):
            raise ValueError(
                "RMS-envelope window sizes "
                "must be unique."
            )

        if config.hop_ratio <= 0:
            raise ValueError(
                "RMS-envelope hop_ratio "
                "must be positive."
            )

        if config.eps <= 0.0:
            raise ValueError(
                "RMS-envelope eps must be positive."
            )

        self.config = config

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> MultiScaleRMSEnvelopeLossOutput:
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target waveform "
                "shapes differ: "
                f"{tuple(prediction.shape)} versus "
                f"{tuple(target.shape)}."
            )

        scale_outputs: list[
            RMSEnvelopeScaleLossOutput
        ] = []

        for window_size in (
            self.config.window_sizes
        ):
            hop_size = max(
                1,
                window_size
                // self.config.hop_ratio,
            )

            prediction_envelope = rms_envelope(
                prediction,
                window_size=window_size,
                hop_size=hop_size,
                eps=self.config.eps,
            )

            target_envelope = rms_envelope(
                target,
                window_size=window_size,
                hop_size=hop_size,
                eps=self.config.eps,
            )

            scale_loss = F.l1_loss(
                prediction_envelope,
                target_envelope,
                reduction="mean",
            )

            scale_outputs.append(
                RMSEnvelopeScaleLossOutput(
                    window_size=window_size,
                    hop_size=hop_size,
                    loss=scale_loss,
                )
            )

        total = torch.stack(
            [
                output.loss
                for output in scale_outputs
            ]
        ).mean()

        return MultiScaleRMSEnvelopeLossOutput(
            total=total,
            scales=tuple(scale_outputs),
        )
