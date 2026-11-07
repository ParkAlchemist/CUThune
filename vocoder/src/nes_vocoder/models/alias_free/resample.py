from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.alias_free.filter import design_lowpass_filter


def _pad_or_crop_to_length(
        x: torch.Tensor,
        target_length: int,
) -> torch.Tensor:
    """
    Symmetrically pad or crop the final dimension to target_length
    """
    current_length = x.shape[-1]
    difference = target_length - current_length

    if difference == 0:
        return x

    if difference > 0:
        pad_left = difference // 2
        pad_right = difference - pad_left
        return F.pad(x, (pad_left, pad_right))

    crop = -difference
    crop_left = crop // 2
    crop_right = crop - crop_left

    end = current_length - crop_right

    return x[..., crop_left:end]


class LowPassFilter1d(nn.Module):
    """
    Fixed depthwise low-pass filter.

    Input & Output:
        [batch, channels, time]
    """

    def __init__(
            self,
            *,
            kernel: torch.Tensor,
    ) -> None:
        super().__init__()

        if kernel.ndim != 1:
            raise ValueError(f"Kernel must be one dimensional, got {tuple(kernel.shape)}")

        if kernel.numel() < 2:
            raise ValueError(f"Kernel must contain at least two taps.")

        self.kernel_size = int(kernel.numel())

        self.register_buffer('kernel', kernel.detach().clone().reshape(1, 1, -1))

    def forward(
            self,
            x: torch.Tensor,
            *,
            target_length: int | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"LowPassFilter1d expects [B, C, T], got {tuple(x.shape)}")

        channels = x.shape[1]

        kernel = self.kernel.to(
            device=x.device,
            dtype=x.dtype,
        ).expand(channels, 1, self.kernel_size)

        # Explicit asymmetric padding supports both even and odd
        # filters while keeping alignment under length contract.
        total_padding = self.kernel_size - 1
        pad_left = total_padding // 2
        pad_right = total_padding - pad_left

        x = F.pad(
            x,
            (pad_left, pad_right),
            mode="reflect",
        )

        y = F.conv1d(
            x,
            kernel,
            stride=1,
            padding=0,
            groups=channels,
        )

        if target_length is not None:
            y = _pad_or_crop_to_length(y, target_length)

        return y


class UpSample1d(nn.Module):
    """
    Insert zeros and apply a reconstruction low-pass filter.

    Input:
        [B, C, T]

    Output:
        [B, C, T * ratio]
    """

    def __init__(
            self,
            *,
            ratio: int = 2,
            kernel_size: int = 12,
    ) -> None:
        super().__init__()

        if ratio < 1:
            raise ValueError(f"Ratio must be positive, got {ratio}")

        self.ratio = ratio
        self.kernel_size = kernel_size

        if ratio == 1:
            self.filter = nn.Identity()
            return

        # At the upsampled rate, the original Nyquist boundary is located at 0.5 / ratio cycles per output sample.
        cutoff = 0.5 / ratio

        # Leave a practical transition region below the new Nyquist
        transition = min(cutoff * 0.25, 0.5 - cutoff)

        kernel = design_lowpass_filter(
            cutoff=cutoff,
            transition_bandwidth=transition,
            kernel_size=kernel_size,
        )

        # Zero insertion reduces DC amplitude by ratio.
        # Scaling the reconstruction filter restores the original signal level.
        kernel = kernel * ratio

        self.filter = LowPassFilter1d(kernel=kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Upsample1d expects [B, C, T], got {tuple(x.shape)}")

        if self.ratio == 1:
            return x

        batch, channels, time = x.shape
        output_length = time * self.ratio

        upsampled = x.new_zeros(batch, channels, output_length)

        upsampled[..., :: self.ratio] = x

        return self.filter(upsampled, target_length=output_length)


class DownSample1d(nn.Module):
    """
    Apply an anti-aliasing filter and decimate.

    Input:
        [B, C, T]

    Output:
        [B, C, ceil(T/ratio)]
    """

    def __init__(
            self,
            *,
            ratio: int = 2,
            kernel_size: int = 12,
    ) -> None:
        super().__init__()

        if ratio < 1:
            raise ValueError(f"Ratio must be positive, got {ratio}")

        self.ratio = ratio
        self.kernel_size = kernel_size

        if ratio == 1:
            self.filter = nn.Identity()
            return

        cutoff = 0.5 / ratio

        transition = min(cutoff * 0.25, 0.5 - cutoff)

        kernel = design_lowpass_filter(
            cutoff=cutoff,
            transition_bandwidth=transition,
            kernel_size=kernel_size,
        )

        self.filter = LowPassFilter1d(kernel=kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"DownSample1d expects [B, C, T], got {tuple(x.shape)}")

        if self.ratio == 1:
            return x

        filtered = self.filter(x, target_length=x.shape[-1])

        return filtered[..., :: self.ratio]

