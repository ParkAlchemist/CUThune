from __future__ import annotations

import torch
import torch.nn as nn

from src.models.alias_free.resample import (
    DownSample1d,
    UpSample1d,
)


class AliasFreeActivation1d(nn.Module):
    """
    Evaluate and activation at a temporary higher sample rate.

    Processing:
        filtered upsampling
        -> activation
        -> filtered downsampling
    """

    def __init__(
            self,
            activation: nn.Module,
            *,
            upsample_ratio: int = 2,
            downsample_ratio: int = 2,
            upsample_kernel_size: int = 12,
            downsample_kernel_size: int = 12,
    ) -> None:
        super().__init__()

        if upsample_ratio != downsample_ratio:
            raise ValueError(f"AliasFreeActivation1d currently requires matching "
                             f"upsample and downsample ratios to preserve length. "
                             f"Got {upsample_ratio} and {downsample_ratio}")

        self.activation = activation

        self.upsample = UpSample1d(
            ratio=upsample_ratio,
            kernel_size=upsample_kernel_size,
        )

        self.downsample = DownSample1d(
            ratio=downsample_ratio,
            kernel_size=downsample_kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_length = x.shape[-1]

        x = self.upsample(x)
        x = self.activation(x)
        x = self.downsample(x)

        if x.shape[-1] != target_length:
            raise RuntimeError(f"Alias-free activation changed sequence length. "
                               f"Expected {target_length}, got {x.shape[-1]}")

        return x

