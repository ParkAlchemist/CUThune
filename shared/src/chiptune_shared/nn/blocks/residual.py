from __future__ import annotations

import torch
import torch.nn as nn

from ..model_utils import (
    NormType,
    PaddingMode,
    get_norm_layer,
    get_padding_layer,
)

from .channel_attention import SqueezeExciteBlock


class ResidualBlock(nn.Module):
    """
    CycleGAN-style residual block.

    Keeps spatial size and channel count unchanged:

    x -> Pad -> Conv -> Norm -> ReLU -> Dropout? -> Pad -> Conv -> Norm -> SE? -> +x
    """
    def __init__(
            self,
            channels: int,
            kernel_size: int = 3,
            padding_mode: PaddingMode = "reflect",
            norm: NormType = "instance",
            dropout: float = 0.0,
            bias: bool | None = None,
            norm_affine: bool = True,
            residual_scale: float = 1.0,
            use_se: bool = False,
            se_reduction: int = 16,
    ) -> None:
        super().__init__()

        if bias is None:
            bias = norm == "none"

        padding = kernel_size // 2
        self.residual_scale = residual_scale

        self.block = nn.Sequential(
            get_padding_layer(padding, padding_mode),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=0,
                bias=bias,
            ),
            get_norm_layer(channels, norm=norm, affine=norm_affine),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            get_padding_layer(padding, padding_mode),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=0,
                bias=bias,
            ),
            get_norm_layer(channels, norm=norm, affine=norm_affine),
        )

        self.se = SqueezeExciteBlock(channels=channels, reduction=se_reduction) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.block(x)
        residual = self.se(residual)
        return x + self.residual_scale * residual

