from __future__ import annotations

import math

import torch


def design_lowpass_filter(
        *,
        cutoff: float,
        transition_bandwidth: float,
        kernel_size: int,
        dtype: torch.dtype = torch.float32,
) -> torch.tensor:
    """
    Create a normalized, symmetric low-pass FIR kernel.

    Parameters
    ----------
    cutoff:
    Pass-band edge in cycles per sample.

    The valid digital-frequency interval is [0, 0.5], where
    0.5 is the Nyquist frequency.

    transition_bandwidth:
    Approximate transition width in cycles per sample.

    The stop-band begins near:
    cutoff + transition_bandwidth

    kernel_size:
    Number of FIR taps. Both even and odd tap counts are
    supported. Even-sized kernels are centered between samples.

    dtype:
    Output tensor dtype.

    Returns
    -------
    torch.Tensor
    A one-dimensional FIR kernel with shape [kernel_size].
    """
    if kernel_size < 2:
        raise ValueError(f"Kernel size must be at least 2, got {kernel_size}")

    if not 0.0 < cutoff < 0.5:
        raise ValueError(f"Cutoff must be between 0 and 0.5, got {cutoff}")

    if transition_bandwidth <= 0.0:
        raise ValueError(f"Transition bandwidth must be positive, got {transition_bandwidth}")

    if cutoff + transition_bandwidth > 0.5:
        raise ValueError(f"Cutoff + transition_bandwidth must be less than 0.5, got {cutoff + transition_bandwidth}")

    # This produces integer-centered coordinates for odd kernels
    # and half-sample-centered coordinates for even kernels.
    positions = (
        torch.arange(kernel_size, dtype=torch.float64) - (kernel_size - 1) / 2.0
    )

    # torch.sinc(z) = sin(pi*z) / (pi*z)
    #
    # The impulse response of a low-pass filter with cutoff fc is:
    #   h[n] = 2*fc*sinc(2*fc*n)
    #
    kernel = 2.0 * cutoff * torch.sinc(2.0 * cutoff * positions)

    # A Kaiser window gives controllable and generally stronger
    # stop-band attenuation than a very short Hann-windowed kernel.
    #
    # beta = 8.6 is a practical starting point for audio resampling
    window = torch.kaiser_window(
        kernel_size,
        periodic=False,
        beta=8.6,
        dtype=torch.float64,
    )

    kernel = kernel * window

    kernel_sum = kernel.sum()

    if not torch.isfinite(kernel_sum):
        raise RuntimeError("Low-pass filter normalization sum is not finite")

    if kernel_sum.abs() < 1e-12:
        raise RuntimeError("Low-pass filter normalization sum is too close to zero")

    # Unity DC gain
    kernel = kernel / kernel_sum

    return kernel.to(dtype=dtype)

