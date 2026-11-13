from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from shared.src.chiptune_shared.nn.blocks.conv_norm_act import ConvNormAct
from shared.src.chiptune_shared.nn.model_utils import (
    NormType,
    PaddingMode,
    ActivationType,
)


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
            *,
            norm: NormType = "group",
            activation: ActivationType = "gelu",
            padding_mode: PaddingMode = "cqt",
            padding_val: float = 0.0,
            upsample_mode: Literal["nearest", "bilinear"] = "nearest",
            num_groups: int = 1,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be positive, "
                             f"got {in_channels}")

        if out_channels <= 0:
            raise ValueError("out_channels must be positive, "
                             f"got {out_channels}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode

        if upsample_mode == "bilinear":
            self.upsample = nn.Upsample(
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            )
        elif upsample_mode == "nearest":
            self.upsample = nn.Upsample(
                scale_factor=2.0,
                mode="nearest",
            )
        else:
            raise ValueError(f"Unsupported upsample mode: {upsample_mode!r}")

        self.conv = ConvNormAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode,
            padding_value=padding_val,
            norm=norm,
            activation=activation,
            spectral_norm=False,
            num_groups=num_groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {tuple(x.shape)}")

        if x.shape[1] != self.in_channels:
            raise ValueError(f"UpsampleBlock channel mismatch: "
                             f"Expected input of shape {self.in_channels}, "
                             f"got {x.shape[1]}")

        x = self.upsample(x)
        x = self.conv(x)
        return x

