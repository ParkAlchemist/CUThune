from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn as nn
from torch.nn import functional as F

from src.config.vocoder_config import MRSTFTConfig


@dataclass(frozen=True)
class STFTResolutionLossOutput:
    fft_size: int
    hop_size: int
    win_length: int

    spectral_convergence: torch.Tensor
    log_magnitude: torch.Tensor

    reference_norm_mean: torch.Tensor
    activity_mean: torch.Tensor
    floor_activation_fraction: torch.Tensor

    spectral_convergence_weight: float
    log_magnitude_weight: float

    @property
    def total(self) -> torch.Tensor:
        return (
            self.spectral_convergence_weight
            * self.spectral_convergence
            + self.log_magnitude_weight
            * self.log_magnitude
        )


@dataclass(frozen=True)
class MultiResolutionSTFTLossOutput:
    total: torch.Tensor

    spectral_convergence: torch.Tensor
    log_magnitude: torch.Tensor

    resolutions: tuple[
        STFTResolutionLossOutput,
        ...,
    ]


class SingleResolutionSTFTLoss(nn.Module):
    """
    One STFT resolution loss.

    Computes:
        1. Spectral convergence loss
        2. Log-magnitude L1 loss

    Input:
        real_audio: [B, 1, T] or [B, T]
        fake_audio: [B, 1, T] or [B, T]
    """

    def __init__(
        self,
        fft_size: int,
        hop_size: int,
        win_length: int,
        eps: float = 1e-7,
        reference_floor: float = 1.0,
    ) -> None:
        super().__init__()

        self.fft_size = int(fft_size)
        self.hop_size = int(hop_size)
        self.win_length = int(win_length)
        self.eps = float(eps)
        self.reference_floor = float(reference_floor)

        self.register_buffer(
            "window",
            torch.hann_window(self.win_length),
            persistent=False,
        )

    def _to_2d_audio(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if x.shape[1] != 1:
                raise ValueError(f"Expected channel dimension 1, got {tuple(x.shape)}")
            x = x[:, 0, :]
        elif x.ndim != 2:
            raise ValueError(f"Expected audio [B,1,T] or [B,T], got {tuple(x.shape)}")

        return x

    def _stft_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        # STFT with complex half precision is experimental and can produce unstable
        # gradients under AMP. Always compute STFT loss in FP32.
        autocast_context = (
            torch.amp.autocast(device_type="cuda", enabled=False)
            if x.is_cuda
            else nullcontext()
        )

        with autocast_context:
            x = self._to_2d_audio(x).float()

            window = self.window.to(device=x.device, dtype=torch.float32)

            stft = torch.stft(
                x,
                n_fft=self.fft_size,
                hop_length=self.hop_size,
                win_length=self.win_length,
                window=window,
                center=True,
                return_complex=True,
            )

            magnitude = torch.abs(stft)

        return magnitude

    @staticmethod
    def _stft_spectral_convergence(
            fake_magnitude: torch.Tensor,
            real_magnitude: torch.Tensor,
            *,
            reference_floor: float,
    ) -> tuple[torch.Tensor, ...]:
        if fake_magnitude.shape != real_magnitude.shape:
            raise ValueError(
                "Fake and real magnitudes must have the same shape, "
                f"got {tuple(fake_magnitude.shape)} and "
                f"{tuple(real_magnitude.shape)}."
            )

        if fake_magnitude.ndim != 3:
            raise ValueError(
                "Expected STFT magnitudes [B,F,T], "
                f"got {tuple(fake_magnitude.shape)}."
            )

        if reference_floor <= 0.0:
            raise ValueError(
                "reference_floor must be positive, "
                f"got {reference_floor}."
            )

        difference_norm = torch.linalg.vector_norm(
            real_magnitude - fake_magnitude,
            dim=(-2, -1),
        )

        reference_norm = torch.linalg.vector_norm(
            real_magnitude,
            dim=(-2, -1),
        )

        relative_error = (
                difference_norm
                / reference_norm.clamp_min(
            reference_floor
        )
        )

        activity_weight = (
                reference_norm.detach()
                / reference_floor
        ).clamp(
            min=0.0,
            max=1.0,
        )

        loss = (relative_error * activity_weight).mean()
        reference_norm_mean = reference_norm.detach().mean()
        activity_mean = activity_weight.detach().mean()
        floor_activation_fraction = (reference_norm < reference_floor).float().detach().mean()

        return loss, reference_norm_mean, activity_mean, floor_activation_fraction

    def forward(
            self,
            fake_audio: torch.Tensor,
            real_audio: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        fake_mag = self._stft_magnitude(fake_audio)
        real_mag = self._stft_magnitude(real_audio)

        spectral_convergence, reference_norm_mean, activity_mean, floor_activation_fraction = self._stft_spectral_convergence(fake_mag, real_mag, reference_floor=self.reference_floor)

        log_magnitude = F.l1_loss(
            torch.log(fake_mag.clamp_min(self.eps)),
            torch.log(real_mag.clamp_min(self.eps)),
        )

        return spectral_convergence, log_magnitude, reference_norm_mean, activity_mean, floor_activation_fraction


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss.

    This is the main non-adversarial waveform reconstruction loss for the
    vocoder. It compares generated and real audio at multiple time-frequency
    resolutions.
    """

    def __init__(
        self,
        config: MRSTFTConfig,
    ) -> None:
        super().__init__()

        if not (
            len(config.fft_sizes)
            == len(config.hop_sizes)
            == len(config.win_lengths)
        ):
            raise ValueError("fft_sizes, hop_sizes, and win_lengths must match length.")

        self.config = config

        self.losses = nn.ModuleList(
            [
                SingleResolutionSTFTLoss(
                    fft_size=fft_size,
                    hop_size=hop_size,
                    win_length=win_length,
                    eps=config.eps,
                )
                for fft_size, hop_size, win_length in zip(
                    config.fft_sizes,
                    config.hop_sizes,
                    config.win_lengths,
                )
            ]
        )

    def forward(
        self,
        fake_audio: torch.Tensor,
        real_audio: torch.Tensor,
    ) -> MultiResolutionSTFTLossOutput:
        resolution_outputs: list[STFTResolutionLossOutput] = []

        for (
                fft_size,
                hop_size,
                win_length,
                loss_func
        ) in zip(
            self.config.fft_sizes,
            self.config.hop_sizes,
            self.config.win_lengths,
            self.losses
        ):
            spectral_convergence, log_magnitude, reference_norm_mean, activity_mean, floor_activation_fraction = loss_func(fake_audio, real_audio)

            resolution_outputs.append(
                STFTResolutionLossOutput(
                    fft_size=fft_size,
                    hop_size=hop_size,
                    win_length=win_length,
                    spectral_convergence=spectral_convergence,
                    log_magnitude=log_magnitude,
                    spectral_convergence_weight=self.config.spectral_convergence_weight,
                    log_magnitude_weight=self.config.log_magnitude_weight,
                    reference_norm_mean=reference_norm_mean,
                    activity_mean=activity_mean,
                    floor_activation_fraction=floor_activation_fraction,
                )
            )

        if not resolution_outputs:
            raise RuntimeError(
                "MultiResolutionSTFTLoss contains no configured resolutions."
            )

        mean_spectral_convergence = torch.stack(
            [
                output.spectral_convergence for output in resolution_outputs
            ]
        ).mean()

        mean_log_magnitude = torch.stack(
            [
                output.log_magnitude for output in resolution_outputs
            ]
        ).mean()

        total = (
            self.config.spectral_convergence_weight * mean_spectral_convergence
            + self.config.log_magnitude_weight * mean_log_magnitude
        )

        return MultiResolutionSTFTLossOutput(
            total=total,
            spectral_convergence=mean_spectral_convergence,
            log_magnitude=mean_log_magnitude,
            resolutions=tuple(resolution_outputs),
        )
