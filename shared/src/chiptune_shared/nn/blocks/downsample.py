from __future__ import annotations

import torch
import torch.nn as nn

from ..model_utils import (
    NormType,
    PaddingMode,
    ActivationType,
)

from ..blocks.conv_norm_act import ConvNormAct


class DownsampleBlock(nn.Module):
    """
    Strided convolution downsampling block.

    For CQT data:
    [B, C, 96, 172] -> [B, 2C, 48, 86]
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            norm: NormType = "instance",
            activation: ActivationType = "relu",
            padding_mode: PaddingMode = "reflect",
            dropout: float = 0.0,
            spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        self.block = ConvNormAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode=padding_mode,
            norm=norm,
            activation=activation,
            dropout=dropout,
            spectral_norm=spectral_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

