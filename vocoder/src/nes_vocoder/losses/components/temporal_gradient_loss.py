from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn as nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SlewEnvelopeScaleLossOutput:
    lag: int
    window_size: int
    hop_size: int

    mean: torch.Tensor
    peak: torch.Tensor
    mean_weight: float
    peak_weight: float

    @property
    def total(self) -> torch.Tensor:
        return (
            self.mean_weight * self.mean
            + self.peak_weight * self.peak
        )


@dataclass(frozen=True)
class MultiScaleSlewEnvelopeLossOutput:
    total: torch.Tensor

    mean: torch.Tensor
    peak: torch.Tensor

    scales: tuple[
        SlewEnvelopeScaleLossOutput,
        ...,
    ]


def lagged_difference(
        waveform: torch.Tensor,
        lag: int,
) -> torch.Tensor:
    if waveform.ndim != 3:
        raise ValueError(
            f"Expected waveform [B, C, T], got {tuple(waveform.shape)}"
        )

    if lag <= 0:
        raise ValueError(
            f"lag must be positive, got {lag}"
        )

    if waveform.shape[-1] <= lag:
        raise ValueError(
            f"Waveform must be longer than lag, got Length={waveform.shape[-1]}, lag={lag}"
        )

    return waveform[..., lag:] - waveform[..., :-lag]


def lagged_slew(
    waveform: torch.Tensor,
    *,
    lag: int,
) -> torch.Tensor:
    """
    Absolute lagged waveform derivative.

    Input:
        Waveform: [B,C,T]

    Output:
        [B,C,T-lag]
    """
    if waveform.ndim != 3:
        raise ValueError(
            "Expected waveform [B,C,T], "
            f"got {tuple(waveform.shape)}."
        )

    if lag <= 0:
        raise ValueError(
            f"lag must be positive, got {lag}."
        )

    if waveform.shape[-1] <= lag:
        raise ValueError(
            "Waveform length must exceed lag. "
            f"Length={waveform.shape[-1]}, "
            f"lag={lag}."
        )

    difference = (
        waveform[..., lag:]
        - waveform[..., :-lag]
    )

    return (
        difference.abs()
        / float(lag)
    )


def pooled_slew_envelopes(
    slew: torch.Tensor,
    *,
    window_size: int,
    hop_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    if slew.ndim != 3:
        raise ValueError(
            "Expected slew tensor [B,C,T], "
            f"got {tuple(slew.shape)}."
        )

    if window_size <= 0:
        raise ValueError(
            "window_size must be positive, "
            f"got {window_size}."
        )

    if hop_size <= 0:
        raise ValueError(
            "hop_size must be positive, "
            f"got {hop_size}."
        )

    if slew.shape[-1] < window_size:
        raise ValueError(
            "Slew tensor is shorter than the "
            "pooling window. "
            f"Length={slew.shape[-1]}, "
            f"window_size={window_size}."
        )
    mean_envelope = F.avg_pool1d(
        slew,
        kernel_size=window_size,
        stride=hop_size,
        padding=0,
        count_include_pad=False,
    )

    peak_envelope = F.max_pool1d(
        slew,
        kernel_size=window_size,
        stride=hop_size,
        padding=0,
    )

    return (
        mean_envelope,
        peak_envelope,
    )


class MultiScaleTemporalGradientLoss(nn.Module):
    def __init__(
            self,
            *,
            lags: tuple[int, ...] = (1, 2, 4),
            normalization: str = "none",
            scale_floor: float = 1e-3,
            eps: float = 1e-8,
            edge_weighting: bool = False,
            edge_weight: float = 1.0,
            max_edge_weight: float = 5.0,
    ) -> None:
        super().__init__()

        if not lags:
            raise ValueError("TemporalGradientLoss requires at least one lag")

        if any(lag <= 0 for lag in lags):
            raise ValueError("TemporalGradientLoss lags must be positive, "
                             f"got lags={lags}")

        if len(set(lags)) != len(lags):
            raise ValueError(
                f"Temporal gradient lags must be unique, got lags={lags}"
            )

        if normalization not in {"none", "target_magnitude"}:
            raise ValueError(
                f"Unsupported temporal gradient normalization: {normalization!r}"
            )

        if scale_floor <= 0.0:
            raise ValueError(
                f"Scale floor must be positive, got {scale_floor}"
            )

        if eps <= 0.0:
            raise ValueError(
                f"Eps must be positive, got {eps}"
            )

        if edge_weight < 0.0:
            raise ValueError(
                f"Edge weight must be positive, got {edge_weight}"
            )

        if max_edge_weight < 1.0:
            raise ValueError(
                f"max_edge_weight must be at least 1, got {max_edge_weight}"
            )

        self.lags = tuple(sorted(lags))
        self.normalization = normalization
        self.scale_floor = scale_floor
        self.eps = eps
        self.edge_weighting = edge_weighting
        self.edge_weight = edge_weight
        self.max_edge_weight = max_edge_weight

    def forward(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes do not match. "
                f"Prediction: {tuple(prediction.shape)}, target: {tuple(target.shape)}"
            )

        per_scale_losses: list[torch.Tensor] = []

        for lag in self.lags:
            prediction_difference = lagged_difference(prediction, lag)
            target_difference = lagged_difference(target, lag)

            absolute_error = (prediction_difference - target_difference).abs()

            if self.edge_weighting:
                target_scale = target_difference.detach().abs().mean(dim=-1, keepdim=True).clamp_min(max(self.scale_floor, self.eps))
                relative_edge_strength = target_difference.detach().abs() / target_scale

                weights = (1.0 + self.edge_weight * relative_edge_strength).clamp(max=self.max_edge_weight)

                absolute_error = absolute_error * weights

            scale_loss = absolute_error.mean()

            if self.normalization == "target_magnitude":
                reference_scale = target_difference.detach().abs().mean().clamp_min(max(self.scale_floor, self.eps))
                scale_loss = scale_loss / reference_scale

            per_scale_losses.append(scale_loss)

        return torch.stack(per_scale_losses).mean()


class MultiScaleSlewEnvelopeLoss(
    nn.Module
):
    def __init__(
        self,
        *,
        lags: tuple[int, ...] = (
            1,
            2,
            4,
        ),
        window_sizes: tuple[int, ...] = (
            32,
            128,
            256,
        ),
        hop_ratio: int = 4,
        mean_weight: float = 1.0,
        peak_weight: float = 0.5,
    ) -> None:
        super().__init__()

        if not lags:
            raise ValueError(
                "At least one lag is required."
            )

        if any(lag <= 0 for lag in lags):
            raise ValueError(
                "All lags must be positive."
            )

        if len(set(lags)) != len(lags):
            raise ValueError(
                "Lags must be unique."
            )

        if not window_sizes:
            raise ValueError(
                "At least one window size "
                "is required."
            )

        if any(
            window_size <= 0
            for window_size in window_sizes
        ):
            raise ValueError(
                "All window sizes must be positive."
            )

        if len(set(window_sizes)) != len(window_sizes):
            raise ValueError(
                "Window sizes must be unique."
            )

        if hop_ratio <= 0:
            raise ValueError(
                "hop_ratio must be positive."
            )

        if mean_weight < 0.0:
            raise ValueError(
                "mean_weight must be non-negative."
            )

        if peak_weight < 0.0:
            raise ValueError(
                "peak_weight must be non-negative."
            )

        if (
            mean_weight == 0.0
            and peak_weight == 0.0
        ):
            raise ValueError(
                "At least one component weight "
                "must be positive."
            )

        self.lags = tuple(sorted(lags))
        self.window_sizes = tuple(
            sorted(window_sizes)
        )

        self.hop_ratio = hop_ratio
        self.mean_weight = mean_weight
        self.peak_weight = peak_weight

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> MultiScaleSlewEnvelopeLossOutput:
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target waveform "
                "shapes differ: "
                f"{tuple(prediction.shape)} versus "
                f"{tuple(target.shape)}."
            )

        scale_outputs: list[
            SlewEnvelopeScaleLossOutput
        ] = []

        for lag in self.lags:
            prediction_slew = lagged_slew(
                prediction,
                lag=lag,
            )

            target_slew = lagged_slew(
                target,
                lag=lag,
            )

            for window_size in (
                self.window_sizes
            ):
                hop_size = max(
                    1,
                    window_size
                    // self.hop_ratio,
                )

                (
                    prediction_mean,
                    prediction_peak,
                ) = pooled_slew_envelopes(
                    prediction_slew,
                    window_size=window_size,
                    hop_size=hop_size,
                )

                (
                    target_mean,
                    target_peak,
                ) = pooled_slew_envelopes(
                    target_slew,
                    window_size=window_size,
                    hop_size=hop_size,
                )

                mean_loss = F.l1_loss(
                    prediction_mean,
                    target_mean,
                    reduction="mean",
                )

                peak_loss = F.l1_loss(
                    prediction_peak,
                    target_peak,
                    reduction="mean",
                )

                scale_outputs.append(
                    SlewEnvelopeScaleLossOutput(
                        lag=lag,
                        window_size=window_size,
                        hop_size=hop_size,
                        mean=mean_loss,
                        peak=peak_loss,
                        mean_weight=self.mean_weight,
                        peak_weight=self.peak_weight,
                    )
                )

        if not scale_outputs:
            raise RuntimeError(
                "Slew-envelope loss produced no "
                "configured scale outputs."
            )

        mean_loss = torch.stack(
            [
                output.mean
                for output in scale_outputs
            ]
        ).mean()

        peak_loss = torch.stack(
            [
                output.peak
                for output in scale_outputs
            ]
        ).mean()

        total = (
            self.mean_weight * mean_loss
            + self.peak_weight * peak_loss
        )

        return MultiScaleSlewEnvelopeLossOutput(
            total=total,
            mean=mean_loss,
            peak=peak_loss,
            scales=tuple(scale_outputs),
        )
