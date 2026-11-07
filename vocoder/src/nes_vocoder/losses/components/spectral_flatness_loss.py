from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn as nn

from vocoder.src.nes_vocoder.config.schema import SpectralFlatnessLossConfig


@dataclass(frozen=True)
class SpectralFlatnessBandLossOutput:
    lower_frequency: float
    upper_frequency: float

    loss: torch.Tensor

    target_log_flatness: torch.Tensor
    prediction_log_flatness: torch.Tensor

    activity_fraction: torch.Tensor
    floor_activation_fraction: torch.Tensor


@dataclass(frozen=True)
class SubBandSpectralFlatnessLossOutput:
    total: torch.Tensor

    bands: tuple[
        SpectralFlatnessBandLossOutput,
        ...,
    ]


def log_spectral_flatness(
    power: torch.Tensor,
    *,
    frequency_dim: int,
    eps: float,
) -> torch.Tensor:
    """
    Calculate log spectral flatness.

    Input power is expected to be non-negative.

    A value near zero indicates a flat spectrum.
    More negative values indicate increasingly
    concentrated or tonal spectra.
    """
    if eps <= 0.0:
        raise ValueError(
            f"eps must be positive, got {eps}."
        )

    if torch.any(power < 0.0):
        raise ValueError(
            "Spectral-flatness power must be "
            "non-negative."
        )

    mean_log_power = torch.log(
        power + eps
    ).mean(
        dim=frequency_dim,
    )

    log_mean_power = torch.log(
        power.mean(
            dim=frequency_dim,
        )
        + eps
    )

    return (
        mean_log_power
        - log_mean_power
    )


class SubBandSpectralFlatnessLoss(
    nn.Module
):
    def __init__(
        self,
        *,
        sample_rate: int,
        config: SpectralFlatnessLossConfig,
    ) -> None:
        super().__init__()

        if sample_rate <= 0:
            raise ValueError(
                "sample_rate must be positive."
            )

        if config.fft_size <= 0:
            raise ValueError(
                "fft_size must be positive."
            )

        if config.hop_size <= 0:
            raise ValueError(
                "hop_size must be positive."
            )

        if not (
            0 < config.win_length
            <= config.fft_size
        ):
            raise ValueError(
                "Expected 0 < win_length "
                "<= fft_size."
            )

        if len(config.band_edges) < 2:
            raise ValueError(
                "At least two band edges are "
                "required."
            )

        if any(
            right <= left
            for left, right in zip(
                config.band_edges,
                config.band_edges[1:],
            )
        ):
            raise ValueError(
                "Band edges must be strictly "
                "increasing."
            )

        if (
            config.band_edges[-1]
            > sample_rate / 2.0
        ):
            raise ValueError(
                "Band edges must not exceed "
                "Nyquist frequency."
            )

        if config.energy_floor <= 0.0:
            raise ValueError(
                "energy_floor must be positive."
            )

        if config.minimum_bins_per_band < 2:
            raise ValueError(
                "minimum_bins_per_band must be "
                "at least two."
            )

        if config.eps <= 0.0:
            raise ValueError(
                "eps must be positive."
            )

        self.sample_rate = sample_rate
        self.config = config

        self.register_buffer(
            "window",
            torch.hann_window(
                config.win_length,
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.register_buffer(
            "window_energy",
            self.window.square().sum(),
            persistent=False,
        )

        frequencies = torch.linspace(
            0.0,
            sample_rate / 2.0,
            config.fft_size // 2 + 1,
            dtype=torch.float32,
        )

        self.register_buffer(
            "frequencies",
            frequencies,
            persistent=False,
        )

        band_bin_indices: list[
            torch.Tensor
        ] = []

        last_band_index = (
            len(config.band_edges) - 2
        )

        for band_index, (
            lower,
            upper,
        ) in enumerate(
            zip(
                config.band_edges,
                config.band_edges[1:],
            )
        ):
            if band_index == last_band_index:
                selected = (
                    (frequencies >= lower)
                    & (frequencies <= upper)
                )
            else:
                selected = (
                    (frequencies >= lower)
                    & (frequencies < upper)
                )

            indices = torch.nonzero(
                selected,
                as_tuple=False,
            ).flatten()

            if (
                indices.numel()
                < config.minimum_bins_per_band
            ):
                raise ValueError(
                    "Spectral-flatness band "
                    f"[{lower}, {upper}] Hz contains "
                    f"only {indices.numel()} FFT bins; "
                    "increase fft_size, widen the band, "
                    "or reduce minimum_bins_per_band."
                )

            band_bin_indices.append(indices)

        # Plain Python tuple is appropriate because
        # the frequency-bin indices are static metadata.
        self.band_bin_indices = tuple(
            band_bin_indices
        )

    def _power_spectrogram(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        if waveform.ndim != 3:
            raise ValueError(
                "Expected waveform [B,C,T], "
                f"got {tuple(waveform.shape)}."
            )

        batch, channels, time = (
            waveform.shape
        )

        flattened = waveform.reshape(
            batch * channels,
            time,
        )

        autocast_context = (
            torch.amp.autocast(
                device_type="cuda",
                enabled=False,
            )
            if waveform.is_cuda
            else nullcontext()
        )

        with autocast_context:
            flattened = flattened.float()

            window = self.window.to(
                device=flattened.device,
                dtype=torch.float32,
            )

            spectrum = torch.stft(
                flattened,
                n_fft=self.config.fft_size,
                hop_length=(
                    self.config.hop_size
                ),
                win_length=(
                    self.config.win_length
                ),
                window=window,
                center=True,
                return_complex=True,
            )

            power = spectrum.abs().square()

        return power.reshape(
            batch,
            channels,
            power.shape[-2],
            power.shape[-1],
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> SubBandSpectralFlatnessLossOutput:
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target waveform "
                "shapes differ: "
                f"{tuple(prediction.shape)} versus "
                f"{tuple(target.shape)}."
            )

        prediction_power = (
            self._power_spectrogram(
                prediction
            )
        )

        target_power = (
            self._power_spectrogram(
                target
            )
        )

        band_outputs: list[
            SpectralFlatnessBandLossOutput
        ] = []

        for band_index, indices in enumerate(
            self.band_bin_indices
        ):
            indices = indices.to(
                device=prediction_power.device
            )

            prediction_band = torch.index_select(
                prediction_power,
                dim=-2,
                index=indices,
            )

            target_band = torch.index_select(
                target_power,
                dim=-2,
                index=indices,
            )

            prediction_flatness = (
                log_spectral_flatness(
                    prediction_band,
                    frequency_dim=-2,
                    eps=self.config.eps,
                )
            )

            target_flatness = (
                log_spectral_flatness(
                    target_band,
                    frequency_dim=-2,
                    eps=self.config.eps,
                )
            )

            target_band_energy = target_band.mean(dim=-2)

            # Smooth target-derived activity gate.
            # Detaching prevents the gate from becoming
            # part of the optimization target.
            window_energy = self.window_energy.to(
                device=target_band.device,
                dtype=target_band.dtype,
            )

            target_band_activity = target_band_energy.detach() / window_energy.clamp_min(self.config.eps)

            activity_weight = (
                    target_band_activity
                    / (
                            target_band_activity
                            + self.config.energy_floor
                    )
            )

            absolute_error = (
                prediction_flatness
                - target_flatness.detach()
            ).abs()

            band_loss = (
                absolute_error
                * activity_weight
            ).mean()

            lower = (
                self.config.band_edges[
                    band_index
                ]
            )

            upper = (
                self.config.band_edges[
                    band_index + 1
                ]
            )

            diagnostic_denominator = activity_weight.sum().clamp_min(self.config.eps)

            target_flatness_mean = (target_flatness.detach() * activity_weight).sum() / diagnostic_denominator

            prediction_flatness_mean = (prediction_flatness.detach() * activity_weight).sum() / diagnostic_denominator

            floor_activation_fraction = (target_band_activity < self.config.energy_floor).float().detach().mean()

            band_outputs.append(
                SpectralFlatnessBandLossOutput(
                    lower_frequency=lower,
                    upper_frequency=upper,
                    loss=band_loss,
                    target_log_flatness=target_flatness_mean,
                    prediction_log_flatness=prediction_flatness_mean,
                    activity_fraction=(
                        activity_weight
                        .detach()
                        .mean()
                    ),
                    floor_activation_fraction=floor_activation_fraction,
                )
            )

        total = torch.stack(
            [
                band.loss
                for band in band_outputs
            ]
        ).mean()

        return (
            SubBandSpectralFlatnessLossOutput(
                total=total,
                bands=tuple(band_outputs),
            )
        )
