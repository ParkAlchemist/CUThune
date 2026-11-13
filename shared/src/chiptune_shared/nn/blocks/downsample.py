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


class PixelUnshuffle(nn.Module):
    def __init__(
            self,
            *,
            ratio: int = 2,
    ) -> None:
        super().__init__()

        if ratio < 2:
            raise ValueError("ratio must be >= 2, "
                             f"got {ratio}")

        self.ratio = ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise RuntimeError("PixelUnshuffle only supports 4D input, "
                               f"got {tuple(x.shape)}")

        B, C, H, W = x.shape

        if H % self.ratio != 0 or W % self.ratio != 0:
            raise RuntimeError(f"Input width or height not divisible by {self.ratio}, "
                               f"got H={H}, W={W}")

        H_out = H // self.ratio
        W_out = W // self.ratio
        C_out = C * (self.ratio ** 2)

        # [B, C, H, W] -> [B, C, H/r, r, W/r, r]
        x = x.reshape(B, C, H_out, self.ratio, W_out, self.ratio)

        # [B, C, H/r r, W/r, r] -> [B, C, r, r, H/r, W/r]
        x = x.permute(0, 1, 3, 5, 2, 4)

        # [B, C, r, r, H/r, W/r] -> [B, C*r*r, H/r, W/r]
        x = x.reshape(B, C_out, H_out, W_out)

        return x


class DownsampleBlock(nn.Module):
    """
    Spatial downsampling with either pixelunshuffle or strided mode.

    For 2D data:
    [B, C, H, W] -> [B, 2C, H/2, W/2]
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            mode: Literal["pixel_unshuffle", "strided"] = "pixel_unshuffle",
            norm: NormType = "group",
            activation: ActivationType = "gelu",
            padding_mode: PaddingMode = "cqt",
            padding_val: float = 0.0,
            spectral_norm: bool = False,
            ratio: int = 2,
            num_groups: int = 1,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be > 0, "
                             f"got {in_channels}")

        if out_channels <= 0:
            raise ValueError("out_channels must be > 0, "
                             f"got {out_channels}")

        if ratio <= 0:
            raise ValueError("ratio must be > 0, "
                             f"got {ratio}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.mode = mode

        if mode == "pixel_unshuffle":
            if ratio < 2:
                raise ValueError("ratio must be >= 2, "
                                 f"got {ratio}")

            self.downsample = PixelUnshuffle(ratio=ratio)
            expanded_channels = in_channels * ratio * ratio
            self.projection = ConvNormAct(
                in_channels=expanded_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode="zeros",
                padding_value=0.0,
                spectral_norm=spectral_norm,
                activation=activation,
                norm=norm,
                num_groups=num_groups,
            )
        elif mode == "strided":
            if ratio != 2:
                raise ValueError("ratio must be == 2, "
                                 f"got {ratio}")

            self.downsample = nn.Identity()
            self.projection = ConvNormAct(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode=padding_mode,
                padding_value=padding_val,
                norm=norm,
                activation=activation,
                spectral_norm=spectral_norm,
                num_groups=num_groups,
            )
        else:
            raise ValueError(f"Unsupported downsampling mode: {mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise RuntimeError("DownsampleBlock only supports 4D input, "
                               f"got {tuple(x.shape)}")

        if x.shape[1] != self.in_channels:
            raise RuntimeError("DownsampleBlock channel mismatch: "
                               f"expected {self.in_channels}, got {x.shape[1]}")

        x = self.downsample(x)
        x = self.projection(x)
        return x

