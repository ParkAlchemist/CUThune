import pytest

from vocoder.src.nes_vocoder.config.schema import VocoderExperimentConfig
from vocoder.src.nes_vocoder.config.validator import (
    ConfigValidationError,
    validate_vocoder_config,
)


def make_valid_config() -> VocoderExperimentConfig:
    config = VocoderExperimentConfig()
    config.data.chip_cache_root = "C:/cache/chip"
    return config


def test_valid_default_config_passes():
    config = make_valid_config()

    validate_vocoder_config(config)


def test_upsampling_product_must_match_hop():
    config = make_valid_config()
    config.data.hop_length = 256
    config.generator.upsample_rates = (8, 8, 4, 2)

    with pytest.raises(
        ConfigValidationError,
        match="upsample_rates",
    ):
        validate_vocoder_config(config)


def test_valid_high_resolution_config_passes():
    config = make_valid_config()

    config.data.sample_rate = 44100
    config.data.hop_length = 256
    config.data.cqt_bins = 202
    config.data.bins_per_octave = 24
    config.data.segment_frames = 128

    config.generator.upsample_rates = (8, 8, 2, 2)
    config.generator.upsample_kernel_sizes = (
        16,
        16,
        4,
        4,
    )

    validate_vocoder_config(config)


def test_cqt_frequency_above_nyquist_rejected():
    config = make_valid_config()

    config.data.sample_rate = 22050
    config.data.cqt_bins = 108
    config.data.bins_per_octave = 12

    with pytest.raises(
        ConfigValidationError,
        match="Nyquist",
    ):
        validate_vocoder_config(config)


def test_persistent_workers_requires_workers():
    config = make_valid_config()
    config.data.num_workers = 0
    config.data.persistent_workers = True

    with pytest.raises(
        ConfigValidationError,
        match="persistent_workers",
    ):
        validate_vocoder_config(config)


def test_mrstft_array_lengths_must_match():
    config = make_valid_config()
    config.loss.mrstft.fft_sizes = (256, 512)
    config.loss.mrstft.hop_sizes = (64,)
    config.loss.mrstft.win_lengths = (256, 512)

    with pytest.raises(
        ConfigValidationError,
        match="equal lengths",
    ):
        validate_vocoder_config(config)


def test_alias_free_requires_snake_beta():
    config = make_valid_config()
    config.generator.activation = "leaky_relu"
    config.generator.alias_free.enabled = True

    with pytest.raises(
        ConfigValidationError,
        match="snake_beta",
    ):
        validate_vocoder_config(config)


def test_at_least_one_discriminator_required():
    config = make_valid_config()
    config.discriminator.use_mpd = False
    config.discriminator.use_msd = False
    config.discriminator.use_mrd = False

    with pytest.raises(
        ConfigValidationError,
        match="At least one discriminator",
    ):
        validate_vocoder_config(config)


def test_optimizer_beta_must_be_valid():
    config = make_valid_config()
    config.optimizer_generator.beta1 = 1.0

    with pytest.raises(
        ConfigValidationError,
        match="beta1",
    ):
        validate_vocoder_config(config)


def test_validator_reports_multiple_errors():
    config = make_valid_config()

    config.data.batch_size = 0
    config.data.num_workers = 0
    config.data.persistent_workers = True
    config.generator.upsample_rates = (8, 8)
    config.optimizer_generator.lr = -1.0

    with pytest.raises(ConfigValidationError) as exc_info:
        validate_vocoder_config(config)

    message = str(exc_info.value)

    assert "data.batch_size" in message
    assert "persistent_workers" in message
    assert "upsample_rates" in message
    assert "optimizer.generator.lr" in message


