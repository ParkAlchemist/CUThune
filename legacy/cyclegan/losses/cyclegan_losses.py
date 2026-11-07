from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import L1Loss

GANLossMode = Literal["lsgan", "vanilla"]


@dataclass
class CycleGANLossConfig:
    gan_mode: GANLossMode = "lsgan"

    lambda_cycle_x: float = 10.0
    lambda_cycle_y: float = 2.0
    lambda_identity_x: float = 5.0
    lambda_identity_y: float = 5.0
    lambda_chroma: float = 2.0

    cycle_noise_std: float = 0.03
    cycle_noise_enabled: bool = True

    chroma_bins_per_octave: int = 12


class GANLoss(nn.Module):
    """
    GAN target loss.

    lsgan:
    MSE(pred, target)

    vanilla:
    BCEWithLogits(pred, target)
    """

    def __init__(self, mode: GANLossMode = "lsgan",) -> None:
        super().__init__()
        self.mode = mode

        if mode == "lsgan":
            self.loss = nn.MSELoss()
        elif mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported GAN loss mode: {mode}")

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool,) -> torch.Tensor:
        target_value = 1.0 if target_is_real else 0.0
        return torch.full_like(prediction, fill_value=target_value)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target)


def inject_gaussian_noise(
        x: torch.Tensor,
        std: float = 0.03,
        enabled: bool = True,
        clamp_min: float = -1.0,
        clamp_max: float = 1.0,
) -> torch.Tensor:
    """
    Anti-steganography noise for cycle path.

    Should be used during training cycle reconstruction, not normal inference.
    """
    if not enabled or std <= 0.0:
        return x

    noise = torch.randn_like(x) * std
    return torch.clamp(x + noise, clamp_min, clamp_max)


def cqt_to_chroma(
        cqt: torch.Tensor,
        bins_per_octave: int = 12,
) -> torch.Tensor:
    """
    Convert normalized CQT magnitude to simple folded chroma.

    Input:
    cqt: [B, 1, F, T] or [B, F, T]

    Output:
    chroma: [B, 12, T]

    Note:
    CQT is normalized [-1, 1]. For chroma, map to [0, 1] first so silence
    near -1 does not become negative energy.
    """
    if cqt.ndim == 4:
        cqt = cqt[:, 0, :, :]
    elif cqt.ndim != 3:
        raise ValueError(f"Expected CQT shape [B, 1, F, T] or [B, F, T], got {tuple(cqt.shape)}")

    b, f, t = cqt.shape

    usable_bins = (f // bins_per_octave) * bins_per_octave
    if usable_bins == 0:
        raise ValueError(f"Frequency bins {f} smaller than bins_per_octave {bins_per_octave}")

    x = cqt[:, :usable_bins, :]

    # [-1, 1] -> [0, 1]
    x = (x + 1.0) * 0.5
    x = torch.clamp(x, 0.0, 1.0)

    # [B, octaves, 12, T] -> [B, 12, T]
    x = x.view(b, -1, bins_per_octave, t)
    chroma = x.mean(dim=1)

    return chroma


class ChromaConsistencyLoss(nn.Module):
    """
    L1 loss between folded CQT chroma representations.

    Used to encourage generated chip CQT to preserve pitch/key structure from
    poly input.
    """

    def __init__(self, bins_per_octave: int = 12) -> None:
        super().__init__()
        self.bins_per_octave = bins_per_octave

    def forward(self, cqt_a: torch.Tensor, cqt_b: torch.Tensor) -> torch.Tensor:
        chroma_a = cqt_to_chroma(cqt_a, bins_per_octave=self.bins_per_octave)
        chroma_b = cqt_to_chroma(cqt_b, bins_per_octave=self.bins_per_octave)
        return F.l1_loss(chroma_a, chroma_b)


@dataclass
class GeneratorLossOutput:
    total: torch.Tensor
    gan_x_to_y: torch.Tensor
    gan_y_to_x: torch.Tensor
    cycle_x: torch.Tensor
    cycle_y: torch.Tensor
    identity_x: torch.Tensor
    identity_y: torch.Tensor
    chroma: torch.Tensor


@dataclass
class DiscriminatorLossOutput:
    total: torch.Tensor
    real: torch.Tensor
    fake: torch.Tensor


@dataclass
class CycleGANLossBundle:
    gan_loss: GANLoss
    l1_loss: nn.L1Loss
    chroma_loss: ChromaConsistencyLoss
    config: CycleGANLossConfig


def build_cyclegan_loss_bundle(
        config: CycleGANLossConfig = CycleGANLossConfig(),
) -> CycleGANLossBundle:
    return CycleGANLossBundle(
        gan_loss=GANLoss(mode=config.gan_mode),
        l1_loss=nn.L1Loss(),
        chroma_loss=ChromaConsistencyLoss(config.chroma_bins_per_octave),
        config=config,
    )


def compute_generator_losses(
        real_x: torch.Tensor,
        real_y: torch.Tensor,
        fake_x: torch.Tensor,
        fake_y: torch.Tensor,
        rec_x: torch.Tensor,
        rec_y: torch.Tensor,
        id_x: torch.Tensor,
        id_y: torch.Tensor,
        pred_fake_x: torch.Tensor,
        pred_fake_y: torch.Tensor,
        bundle: CycleGANLossBundle,
) -> GeneratorLossOutput:
    cfg = bundle.config

    loss_gan_x_to_y = bundle.gan_loss(pred_fake_y, True)
    loss_gan_y_to_x = bundle.gan_loss(pred_fake_x, True)

    loss_cycle_x = bundle.l1_loss(rec_x, real_x) * cfg.lambda_cycle_x
    loss_cycle_y = bundle.l1_loss(rec_y, real_y) * cfg.lambda_cycle_y

    loss_id_x = bundle.l1_loss(id_x, real_x) * cfg.lambda_identity_x
    loss_id_y = bundle.l1_loss(id_y, real_y) * cfg.lambda_identity_y

    if cfg.lambda_chroma > 0.0:
        loss_chroma = bundle.chroma_loss(real_x, fake_y) * cfg.lambda_chroma
    else:
        loss_chroma = torch.zeros((), device=real_x.device, dtype=real_x.dtype)

    total = (
        loss_gan_x_to_y
        + loss_gan_y_to_x
        + loss_cycle_x
        + loss_cycle_y
        + loss_id_x
        + loss_id_y
        + loss_chroma
    )

    return GeneratorLossOutput(
        total=total,
        gan_x_to_y=loss_gan_x_to_y,
        gan_y_to_x=loss_gan_y_to_x,
        cycle_x=loss_cycle_x,
        cycle_y=loss_cycle_y,
        identity_x=loss_id_x,
        identity_y=loss_id_y,
        chroma=loss_chroma,
    )


def compute_discriminator_loss(
        pred_real: torch.Tensor,
        pred_fake_detached: torch.Tensor,
        bundle: CycleGANLossBundle,
) -> DiscriminatorLossOutput:
    loss_real = bundle.gan_loss(pred_real, True)
    loss_fake = bundle.gan_loss(pred_fake_detached, False)

    total = 0.5 * (loss_real + loss_fake)

    return DiscriminatorLossOutput(
        total=total,
        real=loss_real,
        fake=loss_fake,
    )

