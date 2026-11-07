import torch

from src.training.replay_buffer import ReplayBuffer


def test_replay_buffer_preserves_shape_and_device_cpu():
    buffer = ReplayBuffer(max_size=5)

    x = torch.randn(4, 1, 96, 172)

    y = buffer.push_and_pop(x)

    assert y.shape == x.shape
    assert y.device == x.device
    assert len(buffer) == 4


def test_replay_buffer_can_be_disabled():
    buffer = ReplayBuffer(max_size=0)

    x = torch.randn(4, 1, 96, 172)
    y = buffer.push_and_pop(x)

    assert torch.allclose(x, y)
    assert len(buffer) == 0

