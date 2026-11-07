from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import nn

from vocoder.src.nes_vocoder.config.schema import RailOccupancyLossConfig


@dataclass(frozen=True)
class RailOccupancyScaleLossOutput:
    window_size: int
    hop_size: int

    loss: torch.Tensor

    target_occupancy: torch.Tensor
    prediction_occupancy: torch.Tensor

    target_peak_mean: torch.Tensor
    activity_fraction: torch.Tensor


@dataclass(frozen=True)
class MultiScaleRailOccupancyLossOutput:
    total: torch.Tensor

    target_occupancy: torch.Tensor
    prediction_occupancy: torch.Tensor

    scales: tuple[
        RailOccupancyScaleLossOutput,
        ...,
    ]

class MultiScaleRailOccupancyLoss(nn.Module):
    """
    Match the target-relative local occupancy near waveform
    amplitude extrema.

    The loss is:
    - phase insensitive,
    - polarity invariant,
    - target relative,
    - smoothly activity gated,
    - differentiable with respect to prediction.
    """

    def __init__(
        self,
        config: RailOccupancyLossConfig,
    ) -> None:
        super().__init__()

        if not config.window_sizes:
            raise ValueError(
                "Rail-occupancy loss requires at least "
                "one window size."
            )

        if any(
            window_size <= 0
            for window_size
            in config.window_sizes
        ):
            raise ValueError(
                "All rail-occupancy window sizes "
                "must be positive."
            )

        if (
            len(set(config.window_sizes))
            != len(config.window_sizes)
        ):
            raise ValueError(
                "Rail-occupancy window sizes "
                "must be unique."
            )

        if config.hop_ratio <= 0:
            raise ValueError(
                "Rail-occupancy hop_ratio "
                "must be positive."
            )

        if not 0.0 < config.threshold < 1.0:
            raise ValueError(
                "Rail-occupancy threshold must lie "
                "strictly between zero and one."
            )

        if config.sharpness <= 0.0:
            raise ValueError(
                "Rail-occupancy sharpness "
                "must be positive."
            )
        if config.activity_floor <= 0.0:
            raise ValueError(
                "Rail-occupancy activity_floor "
                "must be positive."
            )

        if config.eps <= 0.0:
            raise ValueError(
                "Rail-occupancy eps must be positive."
            )

        self.window_sizes = tuple(sorted(config.window_sizes))

        self.hop_ratio = config.hop_ratio
        self.threshold = float(config.threshold)
        self.sharpness = float(config.sharpness)
        self.activity_floor = float(config.activity_floor)
        self.eps = float(config.eps)

    def _soft_occupancy(
        self,
        windows: torch.Tensor,
        *,
        target_peak: torch.Tensor,
    ) -> torch.Tensor:
        """
        windows:
            [B,C,N,W]

        target_peak:
            [B,C,N,1]

        returns:
            [B,C,N]
        """
        normalized_magnitude = windows.abs() / target_peak.clamp_min(self.eps)

        soft_rail_indicator = torch.sigmoid(self.sharpness * (normalized_magnitude - self.threshold))

        return soft_rail_indicator.mean(dim=-1)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> MultiScaleRailOccupancyLossOutput:
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
            RailOccupancyScaleLossOutput
        ] = []

        for window_size in self.window_sizes:
            if (
                prediction.shape[-1]
                < window_size
            ):
                raise ValueError(
                    "Waveform is shorter than a "
                    "configured rail-occupancy window. "
                    f"Length={prediction.shape[-1]}, "
                    f"window_size={window_size}."
                )

            hop_size = max(
                1,
                window_size
                // self.hop_ratio,
            )

            prediction_windows = (
                prediction.unfold(
                    dimension=-1,
                    size=window_size,
                    step=hop_size,
                )
            )

            target_windows = target.unfold(
                dimension=-1,
                size=window_size,
                step=hop_size,
            )

            target_reference = target_windows.detach()

            target_peak = target_reference.abs().amax(dim=-1, keepdim=True)

            target_occupancy = self._soft_occupancy(target_reference, target_peak=target_peak)

            prediction_occupancy = self._soft_occupancy(prediction_windows, target_peak=target_peak)

            target_peak_without_window_axis = target_peak.squeeze(-1)

            activity_weight = target_peak_without_window_axis / (target_peak_without_window_axis + self.activity_floor)

            occupancy_error = (prediction_occupancy - target_occupancy).abs()

            activity_denominator = activity_weight.sum().clamp_min(self.eps)

            scale_loss = (occupancy_error * activity_weight).sum() / activity_denominator

            weighted_target_occupancy = (target_occupancy * activity_weight).sum() / activity_denominator

            weighted_prediction_occupancy = (prediction_occupancy * activity_weight).sum() / activity_denominator

            scale_outputs.append(
                RailOccupancyScaleLossOutput(
                    window_size=window_size,
                    hop_size=hop_size,
                    loss=scale_loss,
                    target_occupancy=(
                        weighted_target_occupancy
                        .detach()
                    ),
                    prediction_occupancy=(
                        weighted_prediction_occupancy
                        .detach()
                    ),
                    target_peak_mean=(
                        target_peak
                        .detach()
                        .mean()
                    ),
                    activity_fraction=(
                        activity_weight
                        .detach()
                        .mean()
                    ),
                )
            )

        total = torch.stack(
            [
                output.loss
                for output in scale_outputs
            ]
        ).mean()

        target_occupancy_mean = torch.stack(
            [
                output.target_occupancy
                for output in scale_outputs
            ]
        ).mean()

        prediction_occupancy_mean = torch.stack(
            [
                output.prediction_occupancy
                for output in scale_outputs
            ]
        ).mean()


        return MultiScaleRailOccupancyLossOutput(
            total=total,
            target_occupancy=(
                target_occupancy_mean
            ),
            prediction_occupancy=(
                prediction_occupancy_mean
            ),
            scales=tuple(scale_outputs),
        )

