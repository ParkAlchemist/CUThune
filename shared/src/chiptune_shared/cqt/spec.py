from __future__ import annotations

import torch


def seconds_to_frames(seconds: float, sample_rate: int, hop_length: int) -> int:
    return max(1, int(round(seconds * sample_rate / hop_length)))


def pad_feature_last_dim(
    tensor: torch.Tensor,
    target_frames: int,
    pad_value: float,
) -> torch.Tensor:
    current_frames = tensor.shape[-1]

    if current_frames >= target_frames:
        return tensor

    pad_frames = target_frames - current_frames
    pad_shape = list(tensor.shape)
    pad_shape[-1] = pad_frames

    padding = torch.full(
        pad_shape,
        fill_value=pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )

    return torch.cat([tensor, padding], dim=-1)


def cqt_window_energy(cqt_window: torch.Tensor) -> float:
    """
    CQT is normalized to [-1, 1], where silence / db_min is near -1.
    Energy is measured after mapping [-1, 1] -> [0, 1].
    """
    x = (cqt_window.float() + 1.0) * 0.5
    return float(x.mean().item())
