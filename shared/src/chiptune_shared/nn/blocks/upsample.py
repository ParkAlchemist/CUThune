from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from ..model_utils import (
    NormType,
    PaddingMode,
    ActivationType,
)

from ..blocks.conv_norm_act import ConvNormAct


class UpsampleBlock(nn.Module):
    """
    Upsampling block.

    Uses interpolation + convolution by default instead of ConvTranspose2d to
    reduce checkerboard artifacts.

    Note:
    If input sizes are odd after downsampling, the top-level generator may
    still need final cropping/padding to exactly match the original input.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            norm: NormType = "instance",
            activation: ActivationType = "relu",
            padding_mode: PaddingMode = "reflect",
            dropout: float = 0.0,
            upsample_mode: Literal["nearest", "bilinear"] = "nearest",
    ) -> None:
        super().__init__()

        self.upsample_mode = upsample_mode

        if upsample_mode == "bilinear":
            self.upsample = nn.Upsample(
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            )
        else:
            self.upsample = nn.Upsample(
                scale_factor=2.0,
                mode="nearest",
            )

        self.conv = ConvNormAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode,
            norm=norm,
            activation=activation,
            dropout=dropout,
            spectral_norm=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        return x

