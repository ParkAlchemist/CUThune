from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn as nn
from torch.nn import functional as F

from vocoder.src.nes_vocoder.config.schema import BandPowerLossConfig


@dataclass(frozen=True)
class BandPowerResolutionLossOutput:
    fft_size: int
    hop_size: int
    win_length: int

    absolute: torch.Tensor
    shape: torch.Tensor

    activity_mean: torch.Tensor
    floor_activation_fraction: torch.Tensor

    absolute_weight: float
    shape_weight: float

    @property
    def total(self) -> torch.Tensor:
        return (
            self.absolute_weight
            * self.absolute
            + self.shape_weight
            * self.shape
        )


@dataclass(frozen=True)
class MultiResolutionBandPowerLossOutput:
    total: torch.Tensor

    absolute: torch.Tensor
    shape: torch.Tensor

    resolutions: tuple[
        BandPowerResolutionLossOutput,
        ...,
    ]


def build_log_frequency_filterbank(
        *,
        sample_rate: int,
        fft_size: int,
        number_of_bands: int,
        fmin: float,
        fmax: float,
) -> torch.Tensor:
    if sample_rate <= 0:
        raise ValueError(
            f"sample_rate must be positive, got {sample_rate}"
        )

    if fft_size <= 0:
        raise ValueError(
            f"fft_size must be positive, got {fft_size}"
        )

    if number_of_bands <= 1:
        raise ValueError(
            "number_of_bands must be greater than one, "
            f"got {number_of_bands}"
        )

    nyquist = sample_rate / 2.0

    if not 0.0 < fmin < fmax <= nyquist:
        raise ValueError(
            "Expected 0 < fmin < fmax <= nyquist, got "
            f"fmin={fmin}, fmax={fmax}, nyquist={nyquist}"
        )

    frequencies = torch.linspace(
        0.0,
        nyquist,
        fft_size // 2 + 1,
        dtype=torch.float32,
    )

    minimum_half_width = (
            frequencies[1] - frequencies[0]
    )

    edges = torch.logspace(
        torch.log10(
            torch.tensor(
                fmin,
                dtype=torch.float32,
            )
        ),
        torch.log10(
            torch.tensor(
                fmax,
                dtype=torch.float32,
            )
        ),
        number_of_bands + 2,
        dtype=torch.float32,
    )

    filters: list[torch.Tensor] = []

    for band_index in range(number_of_bands):
        left = edges[band_index]
        center = edges[band_index + 1]
        right = edges[band_index + 2]

        half_width_left = torch.maximum(
            center - left,
            minimum_half_width,
        )

        half_width_right = torch.maximum(
            right - center,
            minimum_half_width,
        )

        adjusted_left = (center - half_width_left)

        adjusted_right = (center + half_width_right)

        rising = (frequencies - adjusted_left) / half_width_left.clamp_min(1e-12)

        falling = (adjusted_right - frequencies) / half_width_right.clamp_min(1e-12)

        triangle = torch.minimum(
            rising,
            falling,
        ).clamp(
            min=0.0,
            max=1.0,
        )

        filters.append(triangle)

    filterbank = torch.stack(
        filters,
        dim=0,
    )

    coverage = filterbank.sum(
        dim=0,
        keepdim=True,
    )

    covered = coverage > 0.0

    filterbank = torch.where(
        covered,
        filterbank
        / coverage.clamp_min(1e-12),
        filterbank,
    )

    row_sums = filterbank.sum(dim=1)

    if torch.any(row_sums <= 0.0):
        empty_indices = (
            torch.nonzero(
                row_sums <= 0.0,
                as_tuple=False,
            )
            .flatten()
            .tolist()
        )

        raise RuntimeError(
            "Log-frequency filter bank contains empty "
            f"bands at indices {empty_indices}."
        )

    return filterbank


def project_power_to_bands(
        power_spectrogram: torch.Tensor,
        filterbank: torch.Tensor,
) -> torch.Tensor:
    """
    Power spectrogram:
        [B, C, F, T]

    filterbank:
        [bands, F]

    output:
        [B, C, bands, T]
    """
    if power_spectrogram.ndim != 4:
        raise ValueError(
            f"Expected power spectrogram [B, C, F, T], got "
            f"{tuple(power_spectrogram.shape)}"
        )

    if filterbank.ndim != 2:
        raise ValueError(
            f"Expected filterbank [bands, F], got "
            f"{tuple(filterbank.shape)}"
        )

    if power_spectrogram.shape[-2] != filterbank.shape[-1]:
        raise ValueError(
            f"Spectrogram and filterbank frequency dimensions differ: "
            f"{power_spectrogram.shape[-2]} versus {filterbank.shape[-1]}."
        )

    filterbank = filterbank.to(
        device=power_spectrogram.device,
        dtype=power_spectrogram.dtype,
    )

    return torch.einsum(
        "bcft,kf->bckt",
        power_spectrogram,
        filterbank,
    )


class SingleResolutionBandPowerLoss(
    nn.Module
):
    def __init__(
        self,
        *,
        sample_rate: int,
        fft_size: int,
        hop_size: int,
        win_length: int,
        number_of_bands: int,
        fmin: float,
        fmax: float,
        absolute_weight: float,
        shape_weight: float,
        eps: float,
        energy_floor: float = 1e-4,
    ) -> None:
        super().__init__()

        if fft_size <= 0:
            raise ValueError(
                "fft_size must be positive."
            )

        if hop_size <= 0:
            raise ValueError(
                "hop_size must be positive."
            )

        if win_length <= 0:
            raise ValueError(
                "win_length must be positive."
            )

        if absolute_weight < 0.0:
            raise ValueError(
                "absolute_weight must be non-negative."
            )

        if shape_weight < 0.0:
            raise ValueError(
                "shape_weight must be non-negative."
            )

        if eps <= 0.0:
            raise ValueError(
                "eps must be positive."
            )

        if win_length > fft_size:
            raise ValueError(
                "win_length must not exceed fft_size."
            )

        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.number_of_bands = (
            number_of_bands
        )
        self.absolute_weight = (
            absolute_weight
        )
        self.shape_weight = shape_weight
        self.eps = eps

        if energy_floor <= 0.0:
            raise ValueError(
                "energy_floor must be positive, "
                f"got {energy_floor}."
            )

        self.energy_floor = float(
            energy_floor
        )

        self.register_buffer(
            "window",
            torch.hann_window(
                win_length,
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.register_buffer(
            "filterbank",
            build_log_frequency_filterbank(
                sample_rate=sample_rate,
                fft_size=fft_size,
                number_of_bands=(
                    number_of_bands
                ),
                fmin=fmin,
                fmax=fmax,
            ),
            persistent=False,
        )

        self.register_buffer(
            "power_normalization",
            (
                    torch.tensor(
                        float(self.fft_size),
                        dtype=torch.float32,
                    )
                    * self.window.square().sum()
            ),
            persistent=False,
        )

    def _band_power(
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

        autocast_context = (
            torch.amp.autocast(
                device_type="cuda",
                enabled=False,
            )
            if waveform.is_cuda
            else nullcontext()
        )

        with autocast_context:
            flattened = (
                waveform
                .float()
                .reshape(
                    batch * channels,
                    time,
                )
            )

            window = self.window.to(
                device=flattened.device,
                dtype=torch.float32,
            )

            spectrum = torch.stft(
                flattened,
                n_fft=self.fft_size,
                hop_length=self.hop_size,
                win_length=self.win_length,
                window=window,
                center=True,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )

            power = spectrum.abs().square()

            power = power.reshape(
                batch,
                channels,
                power.shape[-2],
                power.shape[-1],
            )

            band_power = project_power_to_bands(
                power,
                self.filterbank,
            )

        return band_power

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> BandPowerResolutionLossOutput:
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target waveform "
                "shapes differ: "
                f"{tuple(prediction.shape)} versus "
                f"{tuple(target.shape)}."
            )

        prediction_power = self._band_power(
            prediction
        )

        target_power = self._band_power(
            target
        )

        prediction_log_power = torch.log(
            prediction_power.clamp_min(
                self.eps
            )
        )

        target_log_power = torch.log(
            target_power.clamp_min(
                self.eps
            )
        )

        absolute_loss = F.l1_loss(
            prediction_log_power,
            target_log_power,
            reduction="mean",
        )

        prediction_total_power = (
            prediction_power.sum(
                dim=-2,
                keepdim=True,
            )
        )

        target_total_power = (
            target_power.sum(
                dim=-2,
                keepdim=True,
            )
        )

        prediction_distribution = (
                prediction_power
                / prediction_total_power.clamp_min(
            self.eps
        )
        )

        target_distribution = (
                target_power
                / target_total_power.clamp_min(
            self.eps
        )
        )

        shape_error = (
                torch.log(
                    prediction_distribution
                    + self.eps
                )
                - torch.log(
            target_distribution
            + self.eps
        )
        ).abs().mean(
            dim=-2,
        )

        power_normalization = self.power_normalization.to(device=target_power.device, dtype=target_power.dtype)

        target_activity = target_total_power.detach().squeeze(-2) / power_normalization.clamp_min(self.eps)

        activity_weight = (
                target_activity
                / (
                        target_activity
                        + self.energy_floor
                )
        )

        shape_loss = (
                shape_error
                * activity_weight
        ).mean()

        activity_mean = activity_weight.detach().mean()

        floor_activation_fraction = (target_activity < self.energy_floor).float().detach().mean()

        return BandPowerResolutionLossOutput(
            fft_size=self.fft_size,
            hop_size=self.hop_size,
            win_length=self.win_length,
            absolute=absolute_loss,
            shape=shape_loss,
            absolute_weight=(
                self.absolute_weight
            ),
            shape_weight=self.shape_weight,
            activity_mean=activity_mean,
            floor_activation_fraction=floor_activation_fraction,
        )


class MultiResolutionBandPowerLoss(
    nn.Module
):
    def __init__(
        self,
        *,
        sample_rate: int,
        config: BandPowerLossConfig,
    ) -> None:
        super().__init__()

        effective_fmax = (
            config.fmax
            if config.fmax is not None
            else sample_rate / 2.0
        )

        self.config = config
        self.sample_rate = sample_rate

        self.losses = nn.ModuleList(
            [
                SingleResolutionBandPowerLoss(
                    sample_rate=sample_rate,
                    fft_size=fft_size,
                    hop_size=hop_size,
                    win_length=win_length,
                    number_of_bands=(
                        config.number_of_bands
                    ),
                    fmin=config.fmin,
                    fmax=effective_fmax,
                    absolute_weight=(
                        config.absolute_weight
                    ),
                    shape_weight=(
                        config.shape_weight
                    ),
                    eps=config.eps,
                    energy_floor=config.energy_floor,
                )
                for (
                    fft_size,
                    hop_size,
                    win_length,
                ) in zip(
                    config.fft_sizes,
                    config.hop_sizes,
                    config.win_lengths,
                )
            ]
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> MultiResolutionBandPowerLossOutput:
        resolution_outputs = tuple(
            loss_function(
                prediction,
                target,
            )
            for loss_function in self.losses
        )

        if not resolution_outputs:
            raise RuntimeError(
                "Band-power loss contains no "
                "configured resolutions."
            )

        absolute = torch.stack(
            [
                output.absolute
                for output in resolution_outputs
            ]
        ).mean()

        shape = torch.stack(
            [
                output.shape
                for output in resolution_outputs
            ]
        ).mean()

        total = (
            self.config.absolute_weight
            * absolute
            + self.config.shape_weight
            * shape
        )

        return MultiResolutionBandPowerLossOutput(
            total=total,
            absolute=absolute,
            shape=shape,
            resolutions=resolution_outputs,
        )
