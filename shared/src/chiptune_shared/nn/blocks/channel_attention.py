from __future__ import annotations

import torch
import torch.nn as nn

import math


class SqueezeExciteBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super(SqueezeExciteBlock, self).__init__()

        hidden_channels = max(1, channels // reduction)

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


def compute_eca_kernel_size(
        channels: int,
        *,
        gamma: float = 2.0,
        beta: float = 1.0,
        minimum: int = 3,
) -> int:
    """
    Compute an odd ECA channel-convolution kernel size.

    The returned kernel operates across the channel descriptor,
    not across time or spatial dimensions.

    Parameters
    ------------
    channels:
        Number of input feature channels.

    gamma:
        Controls how slowly the kernel grows with channel count.

    beta:
        Offset used by the logarithmic channel to kernel mapping.

    minimum:
        Smallest permitted odd kernel size.

    Returns
    -----------
    int
        Positive odd kernel size.
    """
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}")

    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")

    if minimum <= 0 or minimum % 2 == 0:
        raise ValueError(f"minimum must be a positive odd integer, got {minimum}")

    value = int(abs(math.log2(channels) / gamma + beta / gamma))

    if value % 2 == 0:
        value += 1

    return max(value, minimum)


class ECAChannelGate(nn.Module):
    """
    Compute channel attention from a pooled [B, C, 1] descriptor.

    The Conv1d operates across the channel dimension by interpreting
    the descriptor as a one-channel sequence of length C.

    Input:
        [B, C, 1]

    Output:
        [B, C, 1]
    """
    def __init__(
            self,
            channels: int,
            *,
            kernel_size: int | None = None,
            gamma: float = 2.0,
            beta: float = 1.0,
            minimum_kernel_size: int = 3,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        if kernel_size is None or kernel_size == 0:
            kernel_size = compute_eca_kernel_size(
                channels,
                gamma=gamma,
                beta=beta,
                minimum=minimum_kernel_size,
            )

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be positive odd integer, got {kernel_size}")

        self.channels = channels
        self.kernel_size = kernel_size

        self.channel_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

        self.gate = nn.Sigmoid()

    def forward(
            self,
            descriptor: torch.Tensor,
    ) -> torch.Tensor:
        if descriptor.ndim != 3:
            raise ValueError(f"ECA channel expects [B, C, 1], got {tuple(descriptor.shape)}")

        if descriptor.shape[1] != self.channels:
            raise ValueError(f"ECA channel mismatch: configured for {self.channels}, got {descriptor.shape[1]}")

        if descriptor.shape[2] != 1:
            raise ValueError(f"The final descriptor dimension must equal 1, got {descriptor.shape[2]}")

        # [B, C, 1] -> [B, 1, C]
        attention = descriptor.transpose(1, 2)

        # Local interaction across neighboring feature channels
        attention = self.channel_conv(attention)

        # [B, 1, C] -> [B, C, 1]
        attention = attention.transpose(1, 2)

        return self.gate(attention)


class ECABlock1d(nn.Module):
    """
    Efficient Channel Attention for [B, C, T] tensors.

    Attention is computed from the globally pooled time dimension.
    The module recalibrates channels but does not mix or attend between temporal positions.
    """
    def __init__(
            self,
            channels: int,
            *,
            kernel_size: int | None = None,
            gamma: float = 2.0,
            beta: float = 1.0,
            minimum_kernel_size: int = 3,
            residual: bool = False,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.residual = residual

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.channel_gate = ECAChannelGate(
            channels=channels,
            kernel_size=kernel_size,
            gamma=gamma,
            beta=beta,
            minimum_kernel_size=minimum_kernel_size,
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"ECA channel expects [B, C, T], got {tuple(x.shape)}")

        if x.shape[1] != self.channels:
            raise ValueError(f"ECABlock1d channel mismatch: configured for {self.channels}, got {x.shape[1]}")

        descriptor = self.pool(x)
        attention = self.channel_gate(descriptor)
        scaled = x * attention

        if self.residual:
            return x + scaled

        return scaled


class ECABlock2d(nn.Module):
    """
    Efficient Channel Attention for [B, C, H, W] tensors.
    """
    def __init__(
            self,
            channels: int,
            *,
            kernel_size: int | None = None,
            gamma: float = 2.0,
            beta: float = 1.0,
            minimum_kernel_size: int = 3,
            residual: bool = False,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.residual = residual

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.channel_gate = ECAChannelGate(
            channels=channels,
            kernel_size=kernel_size,
            gamma=gamma,
            beta=beta,
            minimum_kernel_size=minimum_kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"ECA channel expects [B, C, H, W], got {tuple(x.shape)}")

        if x.shape[1] != self.channels:
            raise ValueError(f"ECABlock2d channel mismatch: configured for {self.channels}, got {x.shape[1]}")

        # [B, C, 1, 1] -> [B, C, 1]
        descriptor = self.pool(x).squeeze(-1)

        attention = self.channel_gate(descriptor)

        # [B, C, 1] -> [B, C, 1, 1]
        attention = attention.unsqueeze(-1)

        scaled = x * attention

        if self.residual:
            return x + scaled

        return scaled

