import torch

from vocoder.src.nes_vocoder.models.generator import (
    VocoderGeneratorModelConfig,
    CQTUHiFiGANGenerator,
)


def test_vocoder_generator_shape_32_frames_leaky_relu():
    model = CQTUHiFiGANGenerator(
        96,
        VocoderGeneratorModelConfig(
            upsample_initial_channel=128,
            upsample_rates=(8, 8, 4, 2),
            upsample_kernel_sizes=(16, 16, 8, 4),
            activation="leaky_relu",
        )
    )

    x = torch.randn(2, 96, 32)
    y = model(x)

    assert y.shape == (2, 1, 32 * 512)
    assert torch.isfinite(y).all()


def test_vocoder_generator_shape_32_frames_snake_beta():
    model = CQTUHiFiGANGenerator(
        96,
        VocoderGeneratorModelConfig(
            upsample_initial_channel=128,
            activation="snake_beta",
        )
    )

    x = torch.randn(2, 96, 32)
    y = model(x)

    assert y.shape == (2, 1, 32 * 512)
    assert torch.isfinite(y).all()


def test_vocoder_generator_accepts_4d_cqt():
    model = CQTUHiFiGANGenerator(
        96,
        VocoderGeneratorModelConfig(
            upsample_initial_channel=128,
        )
    )

    x = torch.randn(2, 1, 96, 16)
    y = model(x)

    assert y.shape == (2, 1, 16 * 512)
    assert torch.isfinite(y).all()

