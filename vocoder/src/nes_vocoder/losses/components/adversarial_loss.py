from __future__ import annotations

from typing import Sequence

import torch

from src.losses.vocoder.feature_matching_loss import DISCRIMINATOR_FAMILIES, FamilyLossOutput, mean_enabled_families


def lsgan_generator_adversarial_loss(
    fake_outputs: Sequence[torch.Tensor],
) -> torch.Tensor:
    """
    LSGAN generator adversarial loss.

    Generator wants fake predictions to be classified as real, target 1.
    """
    losses = [
        torch.mean((1.0 - output) ** 2)
        for output in fake_outputs
    ]

    loss = torch.stack(losses).mean()

    return loss


def generator_family_adversarial_losses(
    discriminator_outputs: dict[str, list],
) -> dict[str, torch.Tensor]:
    family_losses: dict[
        str,
        torch.Tensor,
    ] = {}

    for family in DISCRIMINATOR_FAMILIES:
        fake_outputs = discriminator_outputs[
            f"{family}_fake_outputs"
        ]

        if not fake_outputs:
            continue

        family_losses[family] = (
            lsgan_generator_adversarial_loss(
                fake_outputs
            )
        )

    return family_losses


def family_balanced_generator_adversarial_loss(
    discriminator_outputs: dict[str, list],
) -> FamilyLossOutput:
    family_losses = (
        generator_family_adversarial_losses(
            discriminator_outputs
        )
    )

    total = mean_enabled_families(
        family_losses,
        name="generator adversarial loss",
    )

    return FamilyLossOutput(
        total=total,
        families=family_losses,
    )
