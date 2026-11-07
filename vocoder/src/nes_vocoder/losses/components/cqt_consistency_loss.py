from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from nnAudio.features import CQT1992v2

from vocoder.src.nes_vocoder.config.schema import CQTConsistencyLossConfig


@dataclass(frozen=True)
class CQTConsistencyLossOutput:
    total: torch.Tensor
    log_magnitude: torch.Tensor

    prediction_log_magnitude_mean: torch.Tensor
    target_log_magnitude_mean: torch.Tensor



class CQTLogMagnitudeLoss(nn.Module):
    def __init__(
            self,
            *,
            loss_config: CQTConsistencyLossConfig,
    ) -> None:
        super().__init__()

        analysis_config = loss_config.analysis

        if analysis_config.sample_rate <= 0:
            raise ValueError("sample rate must be greater than 0")

        if analysis_config.hop_length <= 0:
            raise ValueError("hop_length must be greater than 0")

        if analysis_config.fmin <= 0.0:
            raise ValueError("fmin must be greater than 0")

        if analysis_config.number_of_bins <= 0:
            raise ValueError("number_of_bins must be greater than 0")

        if analysis_config.bins_per_octave <= 0:
            raise ValueError("bins_per_octave must be greater than 0")

        if analysis_config.filter_scale <= 0.0:
            raise ValueError("filter_scale must be greater than 0")

        if analysis_config.norm not in {1, 2}:
            raise ValueError("norm must be 1 or 2")

        if loss_config.eps <= 0:
            raise ValueError("eps must be greater than 0")

        if loss_config.distance != "log_magnitude_l1":
            raise ValueError(f"Unsupported distance type, got {loss_config.distance}")

        self.analysis_config = analysis_config
        self.loss_config = loss_config
        self.eps = float(loss_config.eps)

        self.cqt = CQT1992v2(
            sr=analysis_config.sample_rate,
            hop_length=analysis_config.hop_length,
            fmin=analysis_config.fmin,
            fmax=None,
            n_bins=analysis_config.number_of_bins,
            bins_per_octave=analysis_config.bins_per_octave,
            filter_scale=analysis_config.filter_scale,
            norm=analysis_config.norm,
            window=analysis_config.window,
            center=analysis_config.center,
            pad_mode=analysis_config.pad_mode,
            trainable=False,
            output_format="Magnitude",
            verbose=False,
        )

        for parameter in self.cqt.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _validate_waveform(
            waveform: torch.Tensor,
            *,
            name: str,
    ) -> None:
        if waveform.ndim not in {2, 3}:
            raise ValueError(f"{name} must have shape [B, T] or [B, 1, T], "
                             f"got {tuple(waveform.shape)}")

        if waveform.ndim == 3 and waveform.shape[1] != 1:
            raise ValueError(f"{name} must be mono, "
                             f"got {tuple(waveform.shape)}")

    @staticmethod
    def _to_2d_audio(
            waveform: torch.Tensor,
    ) -> torch.Tensor:
        if waveform.ndim == 3:
            return waveform[:, 0, :]

        return waveform

    def _magnitude_cqt(
            self,
            waveform: torch.Tensor,
    ) -> torch.Tensor:
        waveform = self._to_2d_audio(waveform).float()

        magnitude = self.cqt(waveform)

        if torch.is_complex(magnitude):
            magnitude = magnitude.abs()

        if not torch.isfinite(magnitude).all():
            raise RuntimeError(
                "CQT magnitude contains non-finite values."
            )

        if magnitude.ndim != 3:
            raise RuntimeError(
                "Expected CQT magnitude [B, F, T], "
                f"got {tuple(magnitude.shape)}"
            )

        return magnitude

    def forward(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor,
    ) -> CQTConsistencyLossOutput:
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target waveform shapes differ: "
                f"{tuple(prediction.shape)} != {tuple(target.shape)}"
            )

        self._validate_waveform(prediction, name="prediction")
        self._validate_waveform(target, name="target")

        autocast_context = (
            torch.amp.autocast(
                device_type="cuda",
                enabled=False,
            )
            if prediction.is_cuda
            else nullcontext()
        )

        with autocast_context:
            prediction_magnitude = self._magnitude_cqt(prediction)

            with torch.no_grad():
                target_magnitude = self._magnitude_cqt(target)

            if prediction_magnitude.shape != target_magnitude.shape:
                raise RuntimeError(
                    "Prediction and target CQT shapes differ: "
                    f"{tuple(prediction_magnitude.shape)} != {tuple(target_magnitude.shape)}"
                )

            prediction_log_magnitude = torch.log(prediction_magnitude.clamp_min(self.eps))
            target_log_magnitude = torch.log(target_magnitude.clamp_min(self.eps))

            log_magnitude_loss = F.l1_loss(prediction_log_magnitude, target_log_magnitude, reduction="mean")

        return CQTConsistencyLossOutput(
            total=log_magnitude_loss,
            log_magnitude=log_magnitude_loss,
            prediction_log_magnitude_mean=prediction_log_magnitude.detach().mean(),
            target_log_magnitude_mean = target_log_magnitude.detach().mean(),
        )
