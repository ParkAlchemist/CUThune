from __future__ import annotations

import torch
import torch.nn as nn

from ..model_utils import (
    NormType,
    get_norm_layer,
    maybe_spectral_norm,
)


class PatchDiscriminatorBlock(nn.Module):
    """
    Basic PatchGAN discriminator block:

    Conv -> Norm? -> LeakyReLU

    Spectral normalization is included as a configurable option because it can
    stabilize GAN discriminators --> set to False baseline.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 4,
            stride: int = 2,
            padding: int = 1,
            norm: NormType = "instance",
            spectral_norm: bool = False,
            use_norm: bool = True,
            negative_slope: float = 0.2,
    ) -> None:
        super().__init__()

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_norm,
        )

        self.conv = maybe_spectral_norm(conv, spectral_norm)

        self.norm = get_norm_layer(out_channels, norm=norm) if use_norm else nn.Identity()

        self.activation = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

