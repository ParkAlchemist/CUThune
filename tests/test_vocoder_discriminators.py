from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from vocoder.src.nes_vocoder.config.schema import (
    MultiPeriodDiscriminatorConfig,
    MultiResolutionDiscriminatorConfig,
    MultiScaleDiscriminatorConfig,
    PeriodDiscriminatorConfig,
    ResolutionDiscriminatorConfig,
    ScaleDiscriminatorConfig,
    VocoderDiscriminatorConfig,
)
from src.models.vocoder_discriminators import (
    PeriodDiscriminator,
    ResolutionDiscriminator,
    ScaleDiscriminator,
    HiFiGANMultiDiscriminator,
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    MultiScaleDiscriminator,
    count_parameters,
)


@pytest.fixture
def waveform_pair() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1337)

    real = torch.randn(
        2,
        1,
        4096,
    )

    fake = torch.randn(
        2,
        1,
        4096,
        requires_grad=True,
    )

    return real, fake


@pytest.fixture
def small_period_config() -> PeriodDiscriminatorConfig:
    return PeriodDiscriminatorConfig(
        period=3,
        channels=(8, 16, 32),
        kernel_size=5,
        stride=3,
        norm="weight",
        negative_slope=0.2,
    )


@pytest.fixture
def small_mpd_config() -> MultiPeriodDiscriminatorConfig:
    return MultiPeriodDiscriminatorConfig(
        periods=(2, 3, 5),
        channels=(8, 16, 32),
        kernel_size=5,
        stride=3,
        norm="weight",
        negative_slope=0.2,
    )


@pytest.fixture
def small_scale_config() -> ScaleDiscriminatorConfig:
    return ScaleDiscriminatorConfig(
        channels=(8, 16, 32),
        kernel_sizes=(7, 5, 3),
        strides=(1, 2, 2),
        groups=(1, 2, 4),
        norm="weight",
        negative_slope=0.2,
    )


@pytest.fixture
def small_msd_config(
    small_scale_config: ScaleDiscriminatorConfig,
) -> MultiScaleDiscriminatorConfig:
    return MultiScaleDiscriminatorConfig(
        num_scales=2,
        first_discriminator_norm="spectral",
        other_discriminator_norm="weight",
        pool_kernel_size=4,
        pool_stride=2,
        pool_padding=2,
        discriminator=small_scale_config,
    )


@pytest.fixture
def small_resolution_config() -> ResolutionDiscriminatorConfig:
    return ResolutionDiscriminatorConfig(
        channels=(8, 16, 32),
        kernel_sizes=(
            (3, 5),
            (3, 5),
            (3, 3),
        ),
        strides=(
            (1, 2),
            (2, 2),
            (1, 1),
        ),
        norm="weight",
        negative_slope=0.2,
        magnitude_compression="log1p",
        eps=1e-7,
    )


@pytest.fixture
def small_mrd_config(
    small_resolution_config: ResolutionDiscriminatorConfig,
) -> MultiResolutionDiscriminatorConfig:
    return MultiResolutionDiscriminatorConfig(
        resolutions=(
            (128, 32, 128),
            (256, 64, 256),
            (512, 128, 512),
        ),
        discriminator=small_resolution_config,
    )


def assert_finite_parameter_gradients(
    module: nn.Module,
) -> None:
    found_gradient = False

    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue

        if parameter.grad is None:
            raise AssertionError(
                f"Parameter {name!r} received no gradient."
            )

        found_gradient = True

        assert torch.isfinite(parameter.grad).all(), (
            f"Parameter {name!r} has a non-finite gradient."
        )

    assert found_gradient


def assert_family_output_valid(
    output: tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[list[torch.Tensor]],
        list[list[torch.Tensor]],
    ],
    *,
    expected_discriminators: int,
    batch_size: int,
) -> None:
    (
        real_outputs,
        fake_outputs,
        real_feature_maps,
        fake_feature_maps,
    ) = output

    assert len(real_outputs) == expected_discriminators
    assert len(fake_outputs) == expected_discriminators
    assert len(real_feature_maps) == expected_discriminators
    assert len(fake_feature_maps) == expected_discriminators

    for real_prediction, fake_prediction in zip(
        real_outputs,
        fake_outputs,
    ):
        assert real_prediction.ndim == 2
        assert fake_prediction.ndim == 2

        assert real_prediction.shape[0] == batch_size
        assert fake_prediction.shape[0] == batch_size

        assert real_prediction.numel() > 0
        assert fake_prediction.numel() > 0

        assert torch.isfinite(real_prediction).all()
        assert torch.isfinite(fake_prediction).all()

    for real_maps, fake_maps in zip(
        real_feature_maps,
        fake_feature_maps,
    ):
        assert real_maps
        assert fake_maps
        assert len(real_maps) == len(fake_maps)

        for real_map, fake_map in zip(
            real_maps,
            fake_maps,
        ):
            assert real_map.shape == fake_map.shape
            assert real_map.shape[0] == batch_size

            assert torch.isfinite(real_map).all()
            assert torch.isfinite(fake_map).all()


@pytest.mark.parametrize(
    ("period", "waveform_length"),
    [
        (2, 128),
        (3, 128),
        (5, 129),
        (7, 130),
    ],
)
def test_period_reshape_produces_divisible_shape(
    period: int,
    waveform_length: int,
) -> None:
    config = PeriodDiscriminatorConfig(
        period=period,
        channels=(8, 16),
        kernel_size=5,
        stride=3,
        norm="none",
        negative_slope=0.2,
    )

    discriminator = PeriodDiscriminator(config)

    waveform = torch.randn(
        2,
        1,
        waveform_length,
    )

    reshaped = discriminator._reshape_by_period(
        waveform
    )

    assert reshaped.ndim == 4
    assert reshaped.shape[0] == 2
    assert reshaped.shape[1] == 1
    assert reshaped.shape[-1] == period

    reconstructed_length = (
        reshaped.shape[-2] * reshaped.shape[-1]
    )

    assert reconstructed_length >= waveform_length
    assert reconstructed_length % period == 0


def test_period_discriminator_forward_and_backward(
    small_period_config: PeriodDiscriminatorConfig,
) -> None:
    discriminator = PeriodDiscriminator(
        small_period_config
    )

    waveform = torch.randn(
        2,
        1,
        1024,
        requires_grad=True,
    )

    prediction, feature_maps = discriminator(waveform)

    assert prediction.ndim == 2
    assert prediction.shape[0] == 2
    assert prediction.numel() > 0
    assert torch.isfinite(prediction).all()
    assert len(feature_maps) == (
        len(small_period_config.channels) + 1
    )

    loss = prediction.square().mean()
    loss.backward()

    assert waveform.grad is not None
    assert torch.isfinite(waveform.grad).all()

    assert_finite_parameter_gradients(discriminator)


def test_mpd_propagates_nested_configuration() -> None:
    config = MultiPeriodDiscriminatorConfig(
        periods=(2, 3),
        channels=(6, 12, 24),
        kernel_size=7,
        stride=2,
        norm="none",
        negative_slope=0.15,
    )

    mpd = MultiPeriodDiscriminator(config)
    assert len(mpd.discriminators) == 2

    for period, discriminator in zip(
        config.periods,
        mpd.discriminators,
    ):
        assert discriminator.period == period
        assert discriminator.config.channels == (6, 12, 24)
        assert discriminator.config.kernel_size == 7
        assert discriminator.config.stride == 2
        assert discriminator.config.norm == "none"
        assert (
            discriminator.config.negative_slope
            == pytest.approx(0.15)
        )


def test_multi_period_discriminator_outputs(
    waveform_pair: tuple[torch.Tensor, torch.Tensor],
    small_mpd_config: MultiPeriodDiscriminatorConfig,
) -> None:
    real, fake = waveform_pair

    mpd = MultiPeriodDiscriminator(
        small_mpd_config
    )

    output = mpd(real, fake)
    assert_family_output_valid(
        output,
        expected_discriminators=len(small_mpd_config.periods),
        batch_size=real.shape[0],
    )


def test_scale_discriminator_rejects_invalid_groups() -> None:
    config = ScaleDiscriminatorConfig(
        channels=(8, 16),
        kernel_sizes=(7, 5),
        strides=(1, 2),
        groups=(1, 3),
        norm="none",
        negative_slope=0.2,
    )

    with pytest.raises(
        ValueError,
        match="divisible",
    ):
        ScaleDiscriminator(config)


def test_scale_discriminator_forward_and_backward(
    small_scale_config: ScaleDiscriminatorConfig,
) -> None:
    discriminator = ScaleDiscriminator(
        small_scale_config
    )

    waveform = torch.randn(
        2,
        1,
        2048,
        requires_grad=True,
    )

    prediction, feature_maps = discriminator(
        waveform
    )

    assert prediction.ndim == 2
    assert prediction.shape[0] == 2
    assert torch.isfinite(prediction).all()

    assert len(feature_maps) == (
        len(small_scale_config.channels) + 1
    )

    prediction.mean().backward()

    assert waveform.grad is not None
    assert torch.isfinite(waveform.grad).all()

    assert_finite_parameter_gradients(discriminator)


def test_msd_propagates_nested_scale_configuration(
    small_msd_config: MultiScaleDiscriminatorConfig,
) -> None:
    msd = MultiScaleDiscriminator(
        small_msd_config
    )

    assert len(msd.discriminators) == 2

    first = msd.discriminators[0]
    second = msd.discriminators[1]

    assert first.config.channels == (
        small_msd_config.discriminator.channels
    )
    assert first.config.kernel_sizes == (
        small_msd_config.discriminator.kernel_sizes
    )
    assert first.config.strides == (
        small_msd_config.discriminator.strides
    )
    assert first.config.groups == (
        small_msd_config.discriminator.groups
    )

    assert (
        first.config.norm
        == small_msd_config.first_discriminator_norm
    )
    assert (
        second.config.norm
        == small_msd_config.other_discriminator_norm
    )

    pooling = msd.pooling

    if isinstance(pooling.kernel_size, tuple):
        pooling.kernel_size = int(pooling.kernel_size[0])

    if isinstance(pooling.stride, tuple):
        pooling.stride = int(pooling.stride[0])

    if isinstance(pooling.padding, tuple):
        pooling.padding = int(pooling.padding[0])

    assert pooling.kernel_size == (
        small_msd_config.pool_kernel_size
    )
    assert pooling.stride == (
        small_msd_config.pool_stride
    )
    assert pooling.padding == (
        small_msd_config.pool_padding
    )


def test_multi_scale_discriminator_outputs(
    waveform_pair: tuple[torch.Tensor, torch.Tensor],
    small_msd_config: MultiScaleDiscriminatorConfig,
) -> None:
    real, fake = waveform_pair

    msd = MultiScaleDiscriminator(
        small_msd_config
    )

    output = msd(real, fake)

    assert_family_output_valid(
        output,
        expected_discriminators=(
            small_msd_config.num_scales
        ),
        batch_size=real.shape[0],
    )


@pytest.mark.parametrize(
    ("fft_size", "hop_size", "win_length"),
    [
        (128, 32, 128),
        (256, 64, 256),
        (512, 128, 512),
    ],
)
def test_resolution_discriminator_spectrogram_shape(
    fft_size: int,
    hop_size: int,
    win_length: int,
    small_resolution_config: ResolutionDiscriminatorConfig,
) -> None:
    discriminator = ResolutionDiscriminator(
        fft_size=fft_size,
        hop_size=hop_size,
        win_length=win_length,
        config=small_resolution_config,
    )

    waveform = torch.randn(
        2,
        1,
        2048,
    )

    spectrogram = discriminator._spectrogram(
        waveform
    )

    assert spectrogram.ndim == 4
    assert spectrogram.shape[0] == 2
    assert spectrogram.shape[1] == 1
    assert spectrogram.shape[2] == (
        fft_size // 2 + 1
    )
    assert spectrogram.shape[3] > 0

    assert torch.isfinite(spectrogram).all()
    assert torch.all(spectrogram >= 0)


@pytest.mark.parametrize(
    "compression",
    [
        "none",
        "log",
        "log1p",
    ],
)
def test_resolution_discriminator_compression_modes(
    compression: str,
    small_resolution_config: ResolutionDiscriminatorConfig,
) -> None:
    config = deepcopy(small_resolution_config)
    config.magnitude_compression = compression

    discriminator = ResolutionDiscriminator(
        fft_size=256,
        hop_size=64,
        win_length=256,
        config=config,
    )

    waveform = torch.randn(2, 1, 2048)

    spectrogram = discriminator._spectrogram(
        waveform
    )

    assert torch.isfinite(spectrogram).all()


def test_resolution_discriminator_rejects_unknown_compression(
    small_resolution_config: ResolutionDiscriminatorConfig,
) -> None:
    config = deepcopy(small_resolution_config)
    config.magnitude_compression = "invalid"

    discriminator = ResolutionDiscriminator(
        fft_size=256,
        hop_size=64,
        win_length=256,
        config=config,
    )

    waveform = torch.randn(1, 1, 1024)

    with pytest.raises(
        ValueError,
        match="compression",
    ):
        discriminator(waveform)


def test_resolution_discriminator_forward(
    small_resolution_config: ResolutionDiscriminatorConfig,
) -> None:
    discriminator = ResolutionDiscriminator(
        fft_size=256,
        hop_size=64,
        win_length=256,
        config=small_resolution_config,
    )

    waveform = torch.randn(
        2,
        1,
        4096,
    )

    prediction, feature_maps = discriminator(waveform)

    assert prediction.ndim == 2
    assert prediction.shape[0] == 2
    assert prediction.shape[1] > 0
    assert torch.isfinite(prediction).all()

    assert len(feature_maps) == (
        len(small_resolution_config.channels) + 1
    )

    for feature_map in feature_maps:
        assert feature_map.ndim == 4
        assert feature_map.shape[0] == 2
        assert torch.isfinite(feature_map.all())


def test_resolution_discriminator_backpropagates_to_waveform(
    small_resolution_config: ResolutionDiscriminatorConfig,
) -> None:
    discriminator = ResolutionDiscriminator(
        fft_size=256,
        hop_size=64,
        win_length=256,
        config=small_resolution_config,
    )

    waveform = torch.randn(
        2,
        1,
        4096,
        requires_grad=True,
    )

    prediction, feature_maps = discriminator(
        waveform
    )

    loss = prediction.square().mean()

    for feature_map in feature_maps:
        loss = loss + 0.001 * feature_map.abs().mean()

    loss.backward()

    assert waveform.grad is not None
    assert torch.isfinite(waveform.grad).all()
    assert waveform.grad.abs().sum() > 0

    assert_finite_parameter_gradients(discriminator)


@pytest.mark.parametrize(
    "compression",
    [
        "none",
        "log",
        "log1p",
    ],
)
def test_resolution_discriminator_handles_silence(
    compression: str,
    small_resolution_config: ResolutionDiscriminatorConfig,
) -> None:
    config = deepcopy(small_resolution_config)
    config.magnitude_compression = compression

    discriminator = ResolutionDiscriminator(
        fft_size=256,
        hop_size=64,
        win_length=256,
        config=config,
    )

    waveform = torch.zeros(
        2,
        1,
        2048,
    )

    prediction, feature_maps = discriminator(
        waveform
    )

    assert torch.isfinite(prediction).all()

    for feature_map in feature_maps:
        assert torch.isfinite(feature_map).all()


def test_resolution_discriminator_rejects_non_mono_input(
    small_resolution_config: ResolutionDiscriminatorConfig,
) -> None:
    discriminator = ResolutionDiscriminator(
        fft_size=256,
        hop_size=64,
        win_length=256,
        config=small_resolution_config,
    )

    waveform = torch.randn(2, 2, 2048)

    with pytest.raises(
        ValueError,
        match="mono",
    ):
        discriminator(waveform)


def test_resolution_discriminator_rejects_wrong_rank(
    small_resolution_config: ResolutionDiscriminatorConfig,
) -> None:
    discriminator = ResolutionDiscriminator(
        fft_size=256,
        hop_size=64,
        win_length=256,
        config=small_resolution_config,
    )

    waveform = torch.randn(2, 2048)

    with pytest.raises(
        ValueError,
        match=r"\[B, 1, T\]",
    ):
        discriminator(waveform)


@pytest.mark.parametrize(
    ("fft_size", "hop_size", "win_length"),
    [
        (0, 64, 256),
        (256, 0, 256),
        (256, 64, 0),
        (256, 64, 512),
    ],
)
def test_resolution_discriminator_rejects_invalid_resolution(
    fft_size: int,
    hop_size: int,
    win_length: int,
    small_resolution_config: ResolutionDiscriminatorConfig,
) -> None:
    with pytest.raises(ValueError):
        ResolutionDiscriminator(
            fft_size=fft_size,
            hop_size=hop_size,
            win_length=win_length,
            config=small_resolution_config,
        )


def test_multi_resolution_discriminator_builds_all_resolutions(
    small_mrd_config: MultiResolutionDiscriminatorConfig,
) -> None:
    mrd = MultiResolutionDiscriminator(
        small_mrd_config
    )

    assert len(mrd.discriminators) == len(
        small_mrd_config.resolutions
    )
    for resolution, discriminator in zip(
        small_mrd_config.resolutions,
        mrd.discriminators,
    ):
        fft_size, hop_size, win_length = resolution

        assert discriminator.fft_size == fft_size
        assert discriminator.hop_size == hop_size
        assert discriminator.win_length == win_length


def test_multi_resolution_discriminator_outputs(
    waveform_pair: tuple[torch.Tensor, torch.Tensor],
    small_mrd_config: MultiResolutionDiscriminatorConfig,
) -> None:
    real, fake = waveform_pair
    mrd = MultiResolutionDiscriminator(
        small_mrd_config
    )

    output = mrd(real, fake)

    assert_family_output_valid(
        output,
        expected_discriminators=len(
            small_mrd_config.resolutions
        ),
        batch_size=real.shape[0],
    )


def test_multi_resolution_discriminator_fake_gradient(
    waveform_pair: tuple[torch.Tensor, torch.Tensor],
    small_mrd_config: MultiResolutionDiscriminatorConfig,
) -> None:
    real, fake = waveform_pair

    mrd = MultiResolutionDiscriminator(
        small_mrd_config
    )

    (
        _,
        fake_outputs,
        _,
        fake_feature_maps,
    ) = mrd(real, fake)

    loss = torch.stack(
        [
            prediction.square().mean()
            for prediction in fake_outputs
        ]
    ).mean()

    feature_losses = []

    for discriminator_maps in fake_feature_maps:
        for feature_map in discriminator_maps:
            feature_losses.append(
                feature_map.abs().mean()
            )

    loss = loss + 0.001 * torch.stack(
        feature_losses
    ).mean()

    loss.backward()

    assert fake.grad is not None
    assert torch.isfinite(fake.grad).all()
    assert fake.grad.abs().sum() > 0

    assert_finite_parameter_gradients(mrd)


def test_multi_resolution_discriminator_rejects_empty_resolutions(
    small_resolution_config: ResolutionDiscriminatorConfig,
) -> None:
    config = MultiResolutionDiscriminatorConfig(
        resolutions=(),
        discriminator=small_resolution_config,
    )

    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        MultiResolutionDiscriminator(config)


def test_multi_resolution_discriminator_rejects_malformed_resolution(
    small_resolution_config: ResolutionDiscriminatorConfig,
) -> None:
    config = MultiResolutionDiscriminatorConfig(
        resolutions=(
            (256, 64),
        ),
        discriminator=small_resolution_config,
    )

    with pytest.raises(
        ValueError,
        match="fft_size",
    ):
        MultiResolutionDiscriminator(config)


def build_combined_config(
    *,
    use_mpd: bool,
    use_msd: bool,
    use_mrd: bool,
    mpd: MultiPeriodDiscriminatorConfig,
    msd: MultiScaleDiscriminatorConfig,
    mrd: MultiResolutionDiscriminatorConfig,
) -> VocoderDiscriminatorConfig:
    return VocoderDiscriminatorConfig(
        use_mpd=use_mpd,
        use_msd=use_msd,
        use_mrd=use_mrd,
        mpd=mpd,
        msd=msd,
        mrd=mrd,
    )


def test_combined_discriminator_requires_one_family(
    small_mpd_config: MultiPeriodDiscriminatorConfig,
    small_msd_config: MultiScaleDiscriminatorConfig,
    small_mrd_config: MultiResolutionDiscriminatorConfig,
) -> None:
    config = build_combined_config(
        use_mpd=False,
        use_msd=False,
        use_mrd=False,
        mpd=small_mpd_config,
        msd=small_msd_config,
        mrd=small_mrd_config,
    )

    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        HiFiGANMultiDiscriminator(config)


def test_combined_mpd_msd_outputs(
    waveform_pair: tuple[torch.Tensor, torch.Tensor],
    small_mpd_config: MultiPeriodDiscriminatorConfig,
    small_msd_config: MultiScaleDiscriminatorConfig,
    small_mrd_config: MultiResolutionDiscriminatorConfig,
) -> None:
    real, fake = waveform_pair

    config = build_combined_config(
        use_mpd=True,
        use_msd=True,
        use_mrd=False,
        mpd=small_mpd_config,
        msd=small_msd_config,
        mrd=small_mrd_config,
    )

    discriminator = HiFiGANMultiDiscriminator(
        config
    )

    output = discriminator(real, fake)

    expected_count = len(small_mpd_config.periods) + small_msd_config.num_scales

    assert len(output["real_outputs"]) == expected_count
    assert len(output["fake_outputs"]) == expected_count
    assert len(output["real_feature_maps"]) == expected_count
    assert len(output["fake_feature_maps"]) == expected_count

    assert len(output["mpd_real_outputs"]) == len(
        small_mpd_config.periods
    )
    assert len(output["msd_real_outputs"]) == (
        small_msd_config.num_scales
    )
    assert output["mrd_real_outputs"] == []

    assert discriminator.mpd is not None
    assert discriminator.msd is not None
    assert discriminator.mrd is None


def test_combined_mpd_mrd_outputs(
    waveform_pair: tuple[torch.Tensor, torch.Tensor],
    small_mpd_config: MultiPeriodDiscriminatorConfig,
    small_msd_config: MultiScaleDiscriminatorConfig,
    small_mrd_config: MultiResolutionDiscriminatorConfig,
) -> None:
    real, fake = waveform_pair

    config = build_combined_config(
        use_mpd=True,
        use_msd=False,
        use_mrd=True,
        mpd=small_mpd_config,
        msd=small_msd_config,
        mrd=small_mrd_config,
    )

    discriminator = HiFiGANMultiDiscriminator(
        config
    )

    output = discriminator(real, fake)

    expected_count = (
        len(small_mpd_config.periods)
        + len(small_mrd_config.resolutions)
    )

    assert len(output["real_outputs"]) == expected_count
    assert len(output["fake_outputs"]) == expected_count
    assert len(output["real_feature_maps"]) == expected_count
    assert len(output["fake_feature_maps"]) == expected_count

    assert len(output["mpd_real_outputs"]) == len(
        small_mpd_config.periods
    )
    assert output["msd_real_outputs"] == []
    assert len(output["mrd_real_outputs"]) == len(
        small_mrd_config.resolutions
    )

    assert discriminator.mpd is not None
    assert discriminator.msd is None
    assert discriminator.mrd is not None


def test_combined_all_families(
    waveform_pair: tuple[torch.Tensor, torch.Tensor],
    small_mpd_config: MultiPeriodDiscriminatorConfig,
    small_msd_config: MultiScaleDiscriminatorConfig,
    small_mrd_config: MultiResolutionDiscriminatorConfig,
) -> None:
    real, fake = waveform_pair

    config = build_combined_config(
        use_mpd=True,
        use_msd=True,
        use_mrd=True,
        mpd=small_mpd_config,
        msd=small_msd_config,
        mrd=small_mrd_config,
    )

    discriminator = HiFiGANMultiDiscriminator(
        config
    )

    output = discriminator(real, fake)

    expected_count = (
        len(small_mpd_config.periods)
        + small_msd_config.num_scales
        + len(small_mrd_config.resolutions)
    )

    assert len(output["real_outputs"]) == expected_count
    assert len(output["fake_outputs"]) == expected_count


def test_combined_mpd_mrd_backpropagates_to_fake(
    small_mpd_config: MultiPeriodDiscriminatorConfig,
    small_msd_config: MultiScaleDiscriminatorConfig,
    small_mrd_config: MultiResolutionDiscriminatorConfig,
) -> None:
    config = build_combined_config(
        use_mpd=True,
        use_msd=False,
        use_mrd=True,
        mpd=small_mpd_config,
        msd=small_msd_config,
        mrd=small_mrd_config,
    )

    discriminator = HiFiGANMultiDiscriminator(
        config
    )

    real = torch.randn(2, 1, 4096)
    fake = torch.randn(
        2,
        1,
        4096,
        requires_grad=True,
    )

    output = discriminator(real, fake)

    adversarial = torch.stack(
        [
            prediction.square().mean()
            for prediction in output["fake_outputs"]
        ]
    ).mean()

    feature_matching = []

    for real_layers, fake_layers in zip(
        output["real_feature_maps"],
        output["fake_feature_maps"],
    ):
        for real_layer, fake_layer in zip(
            real_layers,
            fake_layers,
        ):
            feature_matching.append(
                torch.nn.functional.l1_loss(
                    fake_layer,
                    real_layer.detach(),
                )
            )

    loss = adversarial

    if feature_matching:
        loss = loss + torch.stack(
            feature_matching
        ).mean()

    loss.backward()

    assert fake.grad is not None
    assert torch.isfinite(fake.grad).all()
    assert fake.grad.abs().sum() > 0


def test_resolution_discriminator_applies_weight_norm(
    small_resolution_config: ResolutionDiscriminatorConfig,
) -> None:
    config = deepcopy(small_resolution_config)
    config.norm = "weight"

    discriminator = ResolutionDiscriminator(
        fft_size=256,
        hop_size=64,
        win_length=256,
        config=config,
    )

    for convolution in [
        *discriminator.convs,
        discriminator.conv_post,
    ]:
        assert parametrize.is_parametrized(
            convolution,
            "weight",
        )


def test_resolution_discriminator_removes_norm(
    small_resolution_config: ResolutionDiscriminatorConfig,
) -> None:
    config = deepcopy(small_resolution_config)
    config.norm = "weight"

    discriminator = ResolutionDiscriminator(
        fft_size=256,
        hop_size=64,
        win_length=256,
        config=config,
    )

    discriminator.remove_norm()

    for convolution in [
        *discriminator.convs,
        discriminator.conv_post,
    ]:
        assert not parametrize.is_parametrized(
            convolution,
            "weight",
        )


def test_combined_discriminator_removes_all_norms(
    small_mpd_config: MultiPeriodDiscriminatorConfig,
    small_msd_config: MultiScaleDiscriminatorConfig,
    small_mrd_config: MultiResolutionDiscriminatorConfig,
) -> None:
    config = build_combined_config(
        use_mpd=True,
        use_msd=True,
        use_mrd=True,
        mpd=small_mpd_config,
        msd=small_msd_config,
        mrd=small_mrd_config,
    )

    discriminator = HiFiGANMultiDiscriminator(
        config
    )

    discriminator.remove_norm()

    for module in discriminator.modules():
        if isinstance(
            module,
            (nn.Conv1d, nn.Conv2d),
        ):
            assert not parametrize.is_parametrized(
                module,
                "weight",
            )


def test_mrd_state_dict_roundtrip(
    small_mrd_config: MultiResolutionDiscriminatorConfig,
) -> None:
    first = MultiResolutionDiscriminator(
        small_mrd_config
    )

    second = MultiResolutionDiscriminator(
        small_mrd_config
    )

    second.load_state_dict(
        first.state_dict(),
        strict=True,
    )

    real = torch.randn(1, 1, 2048)
    fake = torch.randn(1, 1, 2048)

    first.eval()
    second.eval()

    with torch.no_grad():
        first_output = first(real, fake)
        second_output = second(real, fake)

    for first_predictions, second_predictions in zip(
        first_output[:2],
        second_output[:2],
    ):
        assert len(first_predictions) == len(
            second_predictions
        )

        for first_prediction, second_prediction in zip(
            first_predictions,
            second_predictions,
        ):
            assert torch.allclose(
                first_prediction,
                second_prediction,
                atol=1e-6,
                rtol=1e-5,
            )


def test_combined_discriminator_state_dict_roundtrip(
    small_mpd_config: MultiPeriodDiscriminatorConfig,
    small_msd_config: MultiScaleDiscriminatorConfig,
    small_mrd_config: MultiResolutionDiscriminatorConfig,
) -> None:
    config = build_combined_config(
        use_mpd=True,
        use_msd=False,
        use_mrd=True,
        mpd=small_mpd_config,
        msd=small_msd_config,
        mrd=small_mrd_config,
    )

    first = HiFiGANMultiDiscriminator(config)
    second = HiFiGANMultiDiscriminator(config)

    second.load_state_dict(
        first.state_dict(),
        strict=True,
    )

    real = torch.randn(1, 1, 2048)
    fake = torch.randn(1, 1, 2048)

    first.eval()
    second.eval()

    with torch.no_grad():
        first_output = first(real, fake)
        second_output = second(real, fake)

    assert len(first_output["real_outputs"]) == len(
        second_output["real_outputs"]
    )

    for first_prediction, second_prediction in zip(
        first_output["real_outputs"],
        second_output["real_outputs"],
    ):
        assert torch.allclose(
            first_prediction,
            second_prediction,
            atol=1e-6,
            rtol=1e-5,
        )


def test_parameter_count_is_positive(
    small_mrd_config: MultiResolutionDiscriminatorConfig,
) -> None:
    discriminator = MultiResolutionDiscriminator(
        small_mrd_config
    )

    trainable = count_parameters(
        discriminator,
        trainable_only=True,
    )

    all_parameters = count_parameters(
        discriminator,
        trainable_only=False,
    )

    assert trainable > 0
    assert all_parameters >= trainable


def test_disabled_families_do_not_add_parameters(
    small_mpd_config: MultiPeriodDiscriminatorConfig,
    small_msd_config: MultiScaleDiscriminatorConfig,
    small_mrd_config: MultiResolutionDiscriminatorConfig,
) -> None:
    mpd_msd_config = build_combined_config(
        use_mpd=True,
        use_msd=True,
        use_mrd=False,
        mpd=small_mpd_config,
        msd=small_msd_config,
        mrd=small_mrd_config,
    )

    mpd_mrd_config = build_combined_config(
        use_mpd=True,
        use_msd=False,
        use_mrd=True,
        mpd=small_mpd_config,
        msd=small_msd_config,
        mrd=small_mrd_config,
    )

    mpd_msd = HiFiGANMultiDiscriminator(
        mpd_msd_config
    )

    mpd_mrd = HiFiGANMultiDiscriminator(
        mpd_mrd_config
    )

    assert mpd_msd.mrd is None
    assert mpd_mrd.msd is None

    assert count_parameters(mpd_msd) > 0
    assert count_parameters(mpd_mrd) > 0


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_mrd_moves_to_cuda_and_backpropagates(
    small_mrd_config: MultiResolutionDiscriminatorConfig,
) -> None:
    device = torch.device("cuda")

    discriminator = MultiResolutionDiscriminator(
        small_mrd_config
    ).to(device)

    real = torch.randn(
        1,
        1,
        4096,
        device=device,
    )

    fake = torch.randn(
        1,
        1,
        4096,
        device=device,
        requires_grad=True,
    )

    output = discriminator(real, fake)

    (
        real_outputs,
        fake_outputs,
        real_maps,
        fake_maps,
    ) = output

    for tensor in real_outputs:
        assert tensor.device.type == device.type

    for tensor in fake_outputs:
        assert tensor.device.type == device.type

    for family_maps in real_maps:
        for tensor in family_maps:
            assert tensor.device.type == device.type

    for family_maps in fake_maps:
        for tensor in family_maps:
            assert tensor.device.type == device.type

    loss = torch.stack(
        [
            prediction.mean()
            for prediction in fake_outputs
        ]
    ).mean()

    loss.backward()

    assert fake.grad is not None
    assert fake.grad.device.type == device.type
    assert torch.isfinite(fake.grad).all()

    for parameter in discriminator.parameters():
        assert parameter.device.type == device.type


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_combined_mpd_mrd_runs_on_cuda(
    small_mpd_config: MultiPeriodDiscriminatorConfig,
    small_msd_config: MultiScaleDiscriminatorConfig,
    small_mrd_config: MultiResolutionDiscriminatorConfig,
) -> None:
    device = torch.device("cuda")

    config = build_combined_config(
        use_mpd=True,
        use_msd=False,
        use_mrd=True,
        mpd=small_mpd_config,
        msd=small_msd_config,
        mrd=small_mrd_config,
    )

    discriminator = HiFiGANMultiDiscriminator(
        config
    ).to(device)

    real = torch.randn(
        1,
        1,
        4096,
        device=device,
    )

    fake = torch.randn(
        1,
        1,
        4096,
        device=device,
        requires_grad=True,
    )

    output = discriminator(real, fake)

    loss = torch.stack(
        [
            prediction.square().mean()
            for prediction in output["fake_outputs"]
        ]
    ).mean()

    loss.backward()

    assert fake.grad is not None
    assert torch.isfinite(fake.grad).all()

    for parameter in discriminator.parameters():
        assert parameter.device.type == device.type


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_mrd_under_cuda_autocast(
    small_mrd_config: MultiResolutionDiscriminatorConfig,
) -> None:
    device = torch.device("cuda")

    discriminator = MultiResolutionDiscriminator(
        small_mrd_config
    ).to(device)

    real = torch.randn(
        1,
        1,
        4096,
        device=device,
    )

    fake = torch.randn(
        1,
        1,
        4096,
        device=device,
        requires_grad=True,
    )

    with torch.amp.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
    ):
        output = discriminator(real, fake)

        loss = torch.stack(
            [
                prediction.float().square().mean()
                for prediction in output[1]
            ]
        ).mean()

    loss.backward()

    assert fake.grad is not None
    assert torch.isfinite(fake.grad).all()


def test_resolution_discriminator_short_input_behavior(
    small_resolution_config: ResolutionDiscriminatorConfig,
) -> None:
    discriminator = ResolutionDiscriminator(
        fft_size=512,
        hop_size=128,
        win_length=512,
        config=small_resolution_config,
    )

    waveform = torch.randn(1, 1, 16)

    with pytest.raises(
        (ValueError, RuntimeError),
    ):
        discriminator(waveform)

