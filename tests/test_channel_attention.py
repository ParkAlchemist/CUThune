import pytest
import torch

from vocoder.src.nes_vocoder.config.schema import VocoderGeneratorModelConfig
from src.models.blocks import (
    ECABlock1d,
    ECABlock2d,
    compute_eca_kernel_size, ECAChannelGate,
)
from vocoder.src.nes_vocoder.models.generator import CQTUHiFiGANGenerator


@pytest.mark.parametrize(
    "channels",
    [1, 4, 16, 32, 64, 128, 256, 512, 1024],
)
def test_eca_kernel_size_is_positive_odd(
    channels: int,
) -> None:
    kernel_size = compute_eca_kernel_size(channels)

    assert kernel_size >= 3
    assert kernel_size % 2 == 1


def test_invalid_eca_channels_raise() -> None:
    with pytest.raises(ValueError):
        compute_eca_kernel_size(0)


def test_eca_1d_preserves_shape_and_gradient() -> None:
    module = ECABlock1d(
        channels=16,
        kernel_size=5,
    )

    x = torch.randn(
        2,
        16,
        128,
        requires_grad=True,
    )

    y = module(x)
    y.square().mean().backward()

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    assert module.channel_gate.channel_conv.weight.grad is not None
    assert torch.isfinite(
        module.channel_gate.channel_conv.weight.grad
    ).all()


def test_eca_2d_preserves_shape_and_gradient() -> None:
    module = ECABlock2d(
        channels=32,
        kernel_size=5,
    )

    x = torch.randn(
        2,
        32,
        29,
        64,
        requires_grad=True,
    )

    y = module(x)
    y.square().mean().backward()

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    assert module.channel_gate.channel_conv.weight.grad is not None


def test_eca_uses_adaptive_kernel() -> None:
    module = ECABlock1d(
        channels=256,
        kernel_size=None,
    )

    assert module.channel_gate.kernel_size % 2 == 1
    assert module.channel_gate.kernel_size >= 3


@pytest.mark.parametrize("kernel_size", [2, 4, -1])
def test_invalid_fixed_kernel_raises(
    kernel_size: int,
) -> None:
    with pytest.raises(ValueError):
        ECAChannelGate(
            channels=16,
            kernel_size=kernel_size,
        )


def test_eca_state_dict_roundtrip() -> None:
    first = ECABlock2d(
        channels=16,
        kernel_size=3,
    )

    second = ECABlock2d(
        channels=16,
        kernel_size=3,
    )

    second.load_state_dict(
        first.state_dict(),
        strict=True,
    )

    x = torch.randn(2, 16, 12, 20)

    first.eval()
    second.eval()

    with torch.no_grad():
        first_output = first(x)
        second_output = second(x)

    assert torch.allclose(
        first_output,
        second_output,
        atol=1e-6,
        rtol=1e-5,
    )


def test_vocoder_has_no_eca_when_disabled():
    config = VocoderGeneratorModelConfig()

    config.eca.enabled = False

    model = CQTUHiFiGANGenerator(cqt_bins=116, model_config=config)

    assert not any(
        isinstance(module, ECABlock1d)
        for module in model.modules()
    )


def test_vocoder_uses_eca_after_mrf():
    config = VocoderGeneratorModelConfig()

    config.eca.enabled = True
    config.eca.kernel_size = 3

    model = CQTUHiFiGANGenerator(cqt_bins=116, model_config=config)

    eca_count = sum(
        isinstance(module, ECABlock1d)
        for module in model.modules()
    )

    assert eca_count == len(config.upsample_rates)

    x = torch.randn(1, 116, 8)
    y = model(x)

    assert y.shape == (1, 1, 8 * 512)
    assert torch.isfinite(y).all()


