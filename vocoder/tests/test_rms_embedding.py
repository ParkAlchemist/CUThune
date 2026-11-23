import torch
from torch import nn
from torch.nn.utils.parametrize import is_parametrized

from vocoder.src.nes_vocoder.config.schema import VocoderGeneratorModelConfig
from vocoder.src.nes_vocoder.models.generator import RMSEnvelopeEmbedding, CQTUHiFiGANGenerator


def test_rms_embedding_silence_remains_zero_after_update() -> None:
    module = RMSEnvelopeEmbedding(
        output_channels=32,
        hidden_channels=8,
        kernel_size=7,
        layer_scale_initial=1e-3,
        remap_to_activity=True,
    )

    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=1e-3,
    )

    active_envelope = torch.zeros(
        2,
        1,
        32,
    )

    active_output, _ = module(
        active_envelope
    )

    active_output.square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    silence = torch.full(
        (2, 1, 32),
        -1.0,
    )

    silence_output, _ = module(
        silence
    )

    assert torch.allclose(
        silence_output,
        torch.zeros_like(
            silence_output
        ),
        atol=1e-7,
        rtol=0.0,
    )


def test_generator_output_length_matches_hop() -> None:
    cqt = torch.randn(
        2,
        116,
        128,
    )

    rms = torch.randn(
        2,
        1,
        128,
    ).clamp(
        -1.0,
        1.0,
    )

    generator = CQTUHiFiGANGenerator(
        cqt_bins=116,
        model_config=VocoderGeneratorModelConfig()
    )

    output = generator(
        cqt,
        rms,
    )

    assert output.shape == (
        2,
        1,
        128
        * generator
        .total_upsample_factor,
    )


def test_generator_diagnostic_path_preserves_audio() -> None:
    cqt = torch.randn(
        2,
        116,
        128,
    )

    rms = torch.rand(
        2,
        1,
        128,
    ) * 2.0 - 1.0

    generator = CQTUHiFiGANGenerator(
        cqt_bins=116,
        model_config=VocoderGeneratorModelConfig()
    )

    ordinary = generator(
        cqt,
        rms,
    )

    diagnostic = generator(
        cqt,
        rms,
        return_diagnostics=True,
    )

    assert torch.allclose(
        ordinary,
        diagnostic.audio,
        atol=1e-6,
        rtol=1e-6,
    )

    assert diagnostic.rms_embedding is not None


def test_generator_weight_norm_removal() -> None:
    generator = CQTUHiFiGANGenerator(
        cqt_bins=116,
        model_config=VocoderGeneratorModelConfig()
    )

    generator.remove_weight_norm()

    for module in generator.modules():
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            assert not is_parametrized(
                module,
                "weight",
            )


