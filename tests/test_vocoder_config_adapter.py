import torch

from src.models.vocoder_hifigan import CQTUHiFiGANGenerator


from vocoder.src.nes_vocoder.config.schema import VocoderExperimentConfig
from vocoder.src.nes_vocoder.config.adapter import (
    build_generator_config,
    build_loss_config,
)


def test_generator_adapter_maps_architecture():
    experiment = VocoderExperimentConfig()

    experiment.data.cqt_bins = 216
    experiment.data.hop_length = 256
    experiment.generator.upsample_initial_channel = 256
    experiment.generator.upsample_rates = (8, 8, 2, 2)
    experiment.generator.upsample_kernel_sizes = (
        16,
        16,
        4,
        4,
    )
    experiment.generator.activation = "snake_beta"

    model_config = build_generator_config(experiment)

    assert model_config.upsample_initial_channel == 256
    assert model_config.upsample_rates == (8, 8, 2, 2)
    assert model_config.upsample_kernel_sizes == (
        16,
        16,
        4,
        4,
    )
    assert model_config.activation == "snake_beta"


def test_loss_adapter_maps_mrstft():
    experiment = VocoderExperimentConfig()

    experiment.loss.lambda_mrstft = 30.0
    experiment.loss.mrstft.fft_sizes = (512, 1024)
    experiment.loss.mrstft.hop_sizes = (128, 256)
    experiment.loss.mrstft.win_lengths = (512, 1024)

    loss_config = build_loss_config(experiment)

    assert loss_config.lambda_mrstft == 30.0
    assert loss_config.mrstft.fft_sizes == (512, 1024)
    assert loss_config.mrstft.hop_sizes == (128, 256)


def test_high_resolution_adapted_generator_shape():
    experiment = VocoderExperimentConfig()

    experiment.data.cqt_bins = 216
    experiment.data.hop_length = 256
    experiment.generator.upsample_initial_channel = 64
    experiment.generator.upsample_rates = (8, 8, 2, 2)
    experiment.generator.upsample_kernel_sizes = (
        16,
        16,
        4,
        4,
    )

    model_config = build_generator_config(experiment)
    model = CQTUHiFiGANGenerator(experiment.data.cqt_bins, model_config)

    cqt = torch.randn(1, 216, 8)
    waveform = model(cqt)

    assert waveform.shape == (1, 1, 8 * 256)
    assert torch.isfinite(waveform).all()

