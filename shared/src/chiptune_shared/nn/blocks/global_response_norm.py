from __future__ import annotations

import torch
from torch import nn as nn


class GlobalResponseNorm1d(nn.Module):
    """
    Global Response Normalization Layer for channel last temporal tensors.

    Expected Input:
        [B, T, C]

    GRN computes a global L2 response over time for each channel,
    normalizes responses relative to their channel-wise mean, and
    applies a learned residual modulation:

        y = x + gamma * (x * normalized_response) + beta

    gamma and beta are initialized to zero, so GRN begins as an
    exact identity mapping.
    """
    def __init__(
            self,
            channels: int,
            *,
            eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.zeros(1, 1, channels, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(1, 1, channels, dtype=torch.float32))

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Input tensor must have 3 dimensions, got {tuple(x.shape)}")

        if x.shape[-1] != self.channels:
            raise RuntimeError(f"GlobalResponseNorm1d channel mismatch:"
                               f"expected {self.channels}, got {x.shape[-1]}")

        # Global L2 response over the temporal dimension
        #
        # [B, T, C] -> [B, 1, C]
        global_response = torch.linalg.vector_norm(
            x,
            ord=2,
            dim=1,
            keepdim=True,
        )

        # Normalize each channel response relative to the mean response across channels
        #
        # [B, 1, C] -> [B, 1, C]
        mean_response = global_response.mean(
            dim=-1,
            keepdim=True,
        )

        normalized_response = global_response / mean_response.clamp_min(self.eps)

        return x + self.gamma * (x * normalized_response) + self.beta


class GlobalResponseNorm2d(nn.Module):
    def __init__(
            self,
            channels: int,
            *,
            eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, channels, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, channels, dtype=torch.float32))

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions, got {tuple(x.shape)}")

        if x.shape[-1] != self.channels:
            raise RuntimeError(f"GlobalResponseNorm2d channel mismatch:"
                               f"expected {self.channels}, got {x.shape[-1]}")

        # Global L2 response over both spatial dimension
        #
        # [B, H, W, C] -> [B, 1, 1, C]
        global_response = torch.linalg.vector_norm(
            x,
            ord=2,
            dim=(1, 2),
            keepdim=True,
        )

        # Normalize each channel response relative to the mean response across channels
        #
        # [B, 1, 1, C] -> [B, 1, 1, C]
        mean_response = global_response.mean(
            dim=-1,
            keepdim=True,
        )

        normalized_response = global_response / mean_response.clamp_min(self.eps)

        return x + self.gamma * (x * normalized_response) + self.beta
