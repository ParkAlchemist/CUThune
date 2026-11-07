from __future__ import annotations

import torch
from torch.nn import functional as F


def waveform_loss(
        fake_audio: torch.Tensor,
        real_audio: torch.Tensor,
) -> torch.Tensor:

    return F.l1_loss(input=fake_audio, target=real_audio)
