from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model_utils import (
    maybe_spectral_norm,
)


class SpatialSelfAttention(nn.Module):
    """
    SAGAN-style spatial self-attention over a 2D CQT feature map.

    Input:
    [B, C, H, W]

    Output:
    [B, C, H, W]

    This is intended for the generator bottleneck, where H and W are reduced
    enough that full spatial attention is affordable.
    """

    def __init__(
            self,
            channels: int,
            reduction: int = 8,
            spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        hidden_channels = max(1, channels // reduction)

        self.query = maybe_spectral_norm(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            spectral_norm,
        )

        self.key = maybe_spectral_norm(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            spectral_norm,
        )

        self.value = maybe_spectral_norm(
            nn.Conv2d(channels, channels, kernel_size=1),
            spectral_norm,
        )

        self.out = maybe_spectral_norm(
            nn.Conv2d(channels, channels, kernel_size=1),
            spectral_norm,
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        n = height * width

        q = self.query(x).view(batch, -1, n).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, n)
        v = self.value(x).view(batch, channels, n)

        attention = torch.bmm(q, k)
        attention = attention / max(1.0, k.shape[1] ** 0.5)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        out = self.out(out)

        return x + self.gamma * out

