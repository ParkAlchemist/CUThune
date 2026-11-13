from __future__ import annotations

import torch
import torch.nn as nn

from shared.src.chiptune_shared.nn.model_utils import (
    NormType,
    ActivationType,
    PaddingMode,
    get_activation,
    get_norm_layer,
    get_padding_layer,
    maybe_spectral_norm,
)


class ConvNormAct(nn.Module):
    """
    Generic 2D convolution block:

    Padding -> Conv2d -> Norm -> Activation

    Use this for both generator and discriminator building blocks.

    For generator:
    norm="instance"
    activation="relu"
    padding_mode="reflect"
    spectral_norm=False

    For discriminator:
    norm="instance" or "none"
    activation="leaky_relu"
    padding_mode="zeros" or "reflect"
    spectral_norm=True optionally
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        padding_mode: PaddingMode = "reflect",
        padding_value: float = 0.0,
        norm: NormType = "instance",
        activation: ActivationType = "relu",
        bias: bool | None = None,
        spectral_norm: bool = False,
        norm_affine: bool = True,
        num_groups: int = 32,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        if bias is None:
            bias = norm == "none"

        self.pad = get_padding_layer(padding, padding_mode, padding_value)

        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

        self.conv = maybe_spectral_norm(conv, spectral_norm)

        self.norm = get_norm_layer(
            out_channels,
            norm = norm,
            num_groups = num_groups,
            affine = norm_affine,
        )

        self.activation = get_activation(
            activation,
            negative_slope = negative_slope,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

