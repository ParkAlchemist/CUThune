from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from contextlib import nullcontext
from src.training.replay_buffer import ReplayBuffer

import torch
import torch.nn as nn

from vocoder.src.nes_vocoder.models.src.losses.cyclegan_losses import (
    CycleGANLossBundle,
    compute_generator_losses,
    compute_discriminator_loss,
    inject_gaussian_noise,
)


@dataclass
class CycleGANModels:
    g_x_to_y: nn.Module
    g_y_to_x: nn.Module
    d_x: nn.Module
    d_y: nn.Module


@dataclass
class CycleGANOptimizers:
    g: torch.optim.Optimizer
    d_x: torch.optim.Optimizer
    d_y: torch.optim.Optimizer


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad


def detach_loss_dict(losses: dict[str, torch.Tensor | float]) -> dict[str, float]:
    out: dict[str, float] = {}

    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            out[key] = float(value.detach().cpu().item())
        else:
            out[key] = float(value)

    return out


def cyclegan_train_step(
    batch: dict[str, Any],
    models: CycleGANModels,
    optimizers: CycleGANOptimizers,
    loss_bundle: CycleGANLossBundle,
    device: torch.device,
    grad_clip_norm: float | None = None,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    fake_x_buffer: ReplayBuffer | None = None,
    fake_y_buffer: ReplayBuffer | None = None,
) -> dict[str, float]:
    """
    One full CycleGAN optimization step.

    Domains:
        X = poly
        Y = chip

    Steps:
        1. Train generators G_XtoY and G_YtoX.
        2. Train D_X.
        3. Train D_Y.
    """

    real_x = batch["real_x"].to(device, non_blocking=True)
    real_y = batch["real_y"].to(device, non_blocking=True)

    g_x_to_y = models.g_x_to_y
    g_y_to_x = models.g_y_to_x
    d_x = models.d_x
    d_y = models.d_y

    cfg = loss_bundle.config

    amp_enabled = bool(use_amp and device.type == "cuda")

    if scaler is None and amp_enabled:
        raise ValueError("AMP is enabled but scaler is None.")

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if amp_enabled
        else nullcontext()
    )

    # ------------------------------------------------------------
    # 1. Train generators
    # ------------------------------------------------------------
    set_requires_grad(d_x, False)
    set_requires_grad(d_y, False)

    optimizers.g.zero_grad(set_to_none=True)

    with autocast_ctx:
        fake_y = g_x_to_y(real_x)
        fake_x = g_y_to_x(real_y)

        noisy_fake_y = inject_gaussian_noise(
            fake_y,
            std=cfg.cycle_noise_std,
            enabled=cfg.cycle_noise_enabled,
        )
        noisy_fake_x = inject_gaussian_noise(
            fake_x,
            std=cfg.cycle_noise_std,
            enabled=cfg.cycle_noise_enabled,
        )

        rec_x = g_y_to_x(noisy_fake_y)
        rec_y = g_x_to_y(noisy_fake_x)

        id_x = g_y_to_x(real_x)
        id_y = g_x_to_y(real_y)

        pred_fake_y_for_g = d_y(fake_y)
        pred_fake_x_for_g = d_x(fake_x)

        g_losses = compute_generator_losses(
            real_x=real_x,
            real_y=real_y,
            fake_y=fake_y,
            fake_x=fake_x,
            rec_x=rec_x,
            rec_y=rec_y,
            id_x=id_x,
            id_y=id_y,
            pred_fake_y=pred_fake_y_for_g,
            pred_fake_x=pred_fake_x_for_g,
            bundle=loss_bundle,
        )

    if amp_enabled:
        assert scaler is not None
        scaler.scale(g_losses.total).backward()

        if grad_clip_norm is not None:
            scaler.unscale_(optimizers.g)
            torch.nn.utils.clip_grad_norm_(
                list(g_x_to_y.parameters()) + list(g_y_to_x.parameters()),
                max_norm=grad_clip_norm,
            )

        scaler.step(optimizers.g)
    else:
        g_losses.total.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(g_x_to_y.parameters()) + list(g_y_to_x.parameters()),
                max_norm=grad_clip_norm,
            )

        optimizers.g.step()

    # Prepare fakes for discriminator.
    # Replay buffers are CPU-backed, so they do not add persistent VRAM usage.
    fake_x_for_d = fake_x.detach()
    fake_y_for_d = fake_y.detach()

    if fake_x_buffer is not None:
        fake_x_for_d = fake_x_buffer.push_and_pop(fake_x_for_d)

    if fake_y_buffer is not None:
        fake_y_for_d = fake_y_buffer.push_and_pop(fake_y_for_d)

    # ------------------------------------------------------------
    # 2. Train D_X
    # ------------------------------------------------------------
    set_requires_grad(d_x, True)
    set_requires_grad(d_y, True)

    optimizers.d_x.zero_grad(set_to_none=True)

    with autocast_ctx:
        pred_real_x = d_x(real_x)
        pred_fake_x = d_x(fake_x_for_d)

        d_x_losses = compute_discriminator_loss(
            pred_real=pred_real_x,
            pred_fake_detached=pred_fake_x,
            bundle=loss_bundle,
        )

    if amp_enabled:
        assert scaler is not None
        scaler.scale(d_x_losses.total).backward()

        if grad_clip_norm is not None:
            scaler.unscale_(optimizers.d_x)
            torch.nn.utils.clip_grad_norm_(d_x.parameters(),
                                           max_norm=grad_clip_norm)

        scaler.step(optimizers.d_x)
    else:
        d_x_losses.total.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(d_x.parameters(),
                                           max_norm=grad_clip_norm)

        optimizers.d_x.step()

    # ------------------------------------------------------------
    # 3. Train D_Y
    # ------------------------------------------------------------
    optimizers.d_y.zero_grad(set_to_none=True)

    with autocast_ctx:
        pred_real_y = d_y(real_y)
        pred_fake_y = d_y(fake_y_for_d)

        d_y_losses = compute_discriminator_loss(
            pred_real=pred_real_y,
            pred_fake_detached=pred_fake_y,
            bundle=loss_bundle,
        )

    if amp_enabled:
        assert scaler is not None
        scaler.scale(d_y_losses.total).backward()

        if grad_clip_norm is not None:
            scaler.unscale_(optimizers.d_y)
            torch.nn.utils.clip_grad_norm_(d_y.parameters(),
                                           max_norm=grad_clip_norm)

        scaler.step(optimizers.d_y)
        scaler.update()
    else:
        d_y_losses.total.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(d_y.parameters(),
                                           max_norm=grad_clip_norm)

        optimizers.d_y.step()

    losses = {
        "loss_g_total": g_losses.total,
        "loss_g_gan_x_to_y": g_losses.gan_x_to_y,
        "loss_g_gan_y_to_x": g_losses.gan_y_to_x,
        "loss_g_cycle_x": g_losses.cycle_x,
        "loss_g_cycle_y": g_losses.cycle_y,
        "loss_g_identity_x": g_losses.identity_x,
        "loss_g_identity_y": g_losses.identity_y,
        "loss_g_chroma": g_losses.chroma,

        "loss_d_x_total": d_x_losses.total,
        "loss_d_x_real": d_x_losses.real,
        "loss_d_x_fake": d_x_losses.fake,

        "loss_d_y_total": d_y_losses.total,
        "loss_d_y_real": d_y_losses.real,
        "loss_d_y_fake": d_y_losses.fake,

        "fake_y_min": fake_y.detach().min(),
        "fake_y_max": fake_y.detach().max(),
        "fake_x_min": fake_x.detach().min(),
        "fake_x_max": fake_x.detach().max(),
    }

    return detach_loss_dict(losses)

