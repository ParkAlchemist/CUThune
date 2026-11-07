from __future__ import annotations

import random
from collections import deque

import torch


class ReplayBuffer:
    """
    CPU replay buffer for CycleGAN fake samples.

    Stores detached fake samples on CPU to avoid increasing VRAM usage.
    During discriminator training, returns either the current fake sample
    or a historical fake sample.

    This follows the standard CycleGAN idea of training discriminators
    on a mixture of recent and older generated samples.
    """

    def __init__(
            self,
            max_size: int = 50,
            return_old_probability: float = 0.5,
    ) -> None:
        self.max_size = max_size
        self.return_old_probability = float(return_old_probability)
        self.data: deque[torch.Tensor] = deque(maxlen=self.max_size)

        if self.max_size < 0:
            raise ValueError("ReplayBuffer max_size must be >= 0")

    def __len__(self) -> int:
        return len(self.data)

    def push_and_pop(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
        batch: Tensor [B, C, H, W], usually fake_x.detach() or fake_y.detach()

        Returns:
        Tensor [B, C, H, W] on the same device as input batch.
        """
        if self.max_size == 0:
            return batch.detach()

        device = batch.device
        dtype = batch.dtype

        output = []

        for sample in batch.detach():
            sample_cpu = sample.unsqueeze(0).cpu().float()

            if len(self.data) < self.max_size:
                self.data.append(sample_cpu)
                output.append(sample_cpu)

            else:
                if random.random() < self.return_old_probability:
                    idx = random.randint(0, len(self.data) - 1)

                    old_sample = self.data[idx].clone()
                    self.data[idx] = sample_cpu

                    output.append(old_sample)
                else:
                    output.append(sample_cpu)

        out = torch.cat(output, dim=0)
        return out.to(device=device, dtype=dtype, non_blocking=True)

    def state_dict(self) -> dict:
        return {
            "max_size": self.max_size,
            "return_old_probability": self.return_old_probability,
            "data": list(self.data),
        }

    def load_state_dict(self, state: dict) -> None:
        self.max_size = int(state["max_size"])
        self.return_old_probability = float(state["return_old_probability"])
        self.data = deque(state["data"], maxlen=self.max_size)

