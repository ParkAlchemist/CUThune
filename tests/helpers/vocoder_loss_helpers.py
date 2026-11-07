from __future__ import annotations

from collections.abc import Mapping

import torch

from vocoder.src.nes_vocoder.config.schema import (
    FeatureMatchingLossConfig,
    MRSTFTConfig,
    TemporalGradientLossConfig,
    VocoderLossConfig, SpectralFlatnessLossConfig, CQTAnalysisConfig, CQTConsistencyLossConfig, RailOccupancyLossConfig,
)
from src.losses.vocoder.cqt_consistency_loss import CQTLogMagnitudeLoss, CQTConsistencyLossOutput
from src.losses.vocoder.rail_occupancy_loss import MultiScaleRailOccupancyLoss, MultiScaleRailOccupancyLossOutput, \
    RailOccupancyScaleLossOutput
from src.losses.vocoder_losses import WeightedLossComponent, FamilyDiscriminatorLossOutput, VocoderGeneratorLossOutput, \
    make_weighted_loss_component
from src.losses.vocoder.feature_matching_loss import FamilyLossOutput
from src.losses.vocoder.dc_offset_loss import DCOffsetScaleLossOutput, MultiScaleDCOffsetLossOutput
from src.losses.vocoder.temporal_gradient_loss import MultiScaleSlewEnvelopeLossOutput
from src.losses.vocoder.spectral_flatness_loss import SubBandSpectralFlatnessLossOutput, \
    SubBandSpectralFlatnessLoss
from src.losses.vocoder.band_power_loss import MultiResolutionBandPowerLossOutput
from src.losses.vocoder.rms_envelope_loss import RMSEnvelopeScaleLossOutput, MultiScaleRMSEnvelopeLossOutput
from src.losses.vocoder.mrstft_loss import STFTResolutionLossOutput, MultiResolutionSTFTLossOutput


def scalar_tensor(
    value: float,
    *,
    requires_grad: bool = False,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Construct a scalar float32 tensor for loss-output tests.
    """
    return torch.tensor(
        float(value),
        dtype=torch.float32,
        device=device,
        requires_grad=requires_grad,
    )


def make_test_stft_resolution_output(
    *,
    fft_size: int = 256,
    hop_size: int = 64,
    win_length: int = 256,
    spectral_convergence: float = 0.4,
    log_magnitude: float = 0.6,
    spectral_convergence_weight: float = 1.0,
    log_magnitude_weight: float = 1.0,
    device: torch.device | str = "cpu",
) -> STFTResolutionLossOutput:
    return STFTResolutionLossOutput(
        fft_size=fft_size,
        hop_size=hop_size,
        win_length=win_length,
        spectral_convergence=scalar_tensor(
            spectral_convergence,
            device=device,
        ),
        log_magnitude=scalar_tensor(
            log_magnitude,
            device=device,
        ),
        spectral_convergence_weight=(
            spectral_convergence_weight
        ),
        log_magnitude_weight=(
            log_magnitude_weight
        ),
    )


def make_test_mrstft_output(
    *,
    resolutions: tuple[
        STFTResolutionLossOutput,
        ...,
    ] | None = None,
    spectral_convergence_weight: float = 1.0,
    log_magnitude_weight: float = 1.0,
    device: torch.device | str = "cpu",
) -> MultiResolutionSTFTLossOutput:
    if resolutions is None:
        resolutions = (
            make_test_stft_resolution_output(
                fft_size=256,
                hop_size=64,
                win_length=256,
                spectral_convergence=0.4,
                log_magnitude=0.6,
                spectral_convergence_weight=(
                    spectral_convergence_weight
                ),
                log_magnitude_weight=(
                    log_magnitude_weight
                ),
                device=device,
            ),
            make_test_stft_resolution_output(
                fft_size=512,
                hop_size=128,
                win_length=512,
                spectral_convergence=0.6,
                log_magnitude=0.8,
                spectral_convergence_weight=(
                    spectral_convergence_weight
                ),
                log_magnitude_weight=(
                    log_magnitude_weight
                ),
                device=device,
            ),
        )

    if not resolutions:
        raise ValueError(
            "At least one MR-STFT resolution is required."
        )

    mean_spectral_convergence = torch.stack(
        [
            resolution.spectral_convergence
            for resolution in resolutions
        ]
    ).mean()

    mean_log_magnitude = torch.stack(
        [
            resolution.log_magnitude
            for resolution in resolutions
        ]
    ).mean()

    total = (
        spectral_convergence_weight
        * mean_spectral_convergence
        + log_magnitude_weight
        * mean_log_magnitude
    )

    return MultiResolutionSTFTLossOutput(
        total=total,
        spectral_convergence=(
            mean_spectral_convergence
        ),
        log_magnitude=mean_log_magnitude,
        resolutions=resolutions,
    )


def make_test_rms_envelope_output(
    *,
    device: torch.device | str = "cpu",
) -> MultiScaleRMSEnvelopeLossOutput:
    scales = (
        RMSEnvelopeScaleLossOutput(
            window_size=64,
            hop_size=16,
            loss=scalar_tensor(
                0.2,
                device=device,
            ),
        ),
        RMSEnvelopeScaleLossOutput(
            window_size=256,
            hop_size=64,
            loss=scalar_tensor(
                0.4,
                device=device,
            ),
        ),
    )

    total = torch.stack(
        [
            scale.loss
            for scale in scales
        ]
    ).mean()

    return MultiScaleRMSEnvelopeLossOutput(
        total=total,
        scales=scales,
    )


def make_zero_mrstft_output(
    *,
    device: torch.device | str = "cpu",
) -> MultiResolutionSTFTLossOutput:
    resolution = make_test_stft_resolution_output(
        fft_size=256,
        hop_size=64,
        win_length=256,
        spectral_convergence=0.0,
        log_magnitude=0.0,
        spectral_convergence_weight=1.0,
        log_magnitude_weight=1.0,
        device=device,
    )

    return MultiResolutionSTFTLossOutput(
        total=scalar_tensor(
            0.0,
            device=device,
        ),
        spectral_convergence=scalar_tensor(
            0.0,
            device=device,
        ),
        log_magnitude=scalar_tensor(
            0.0,
            device=device,
        ),
        resolutions=(resolution,),
    )


def make_test_spectral_flatness_loss(
    *,
    sample_rate: int = 8_000,
    fft_size: int = 128,
    hop_size: int = 32,
    win_length: int = 128,
    band_edges: tuple[float, ...] = (
        0.0,
        500.0,
        2_000.0,
        4_000.0,
    ),
    energy_floor: float = 1e-4,
    minimum_bins_per_band: int = 4,
    eps: float = 1e-8,
    device: torch.device | str = "cpu",
) -> SubBandSpectralFlatnessLoss:
    """
    Construct a small spectral-flatness loss for unit tests.

    The default setup uses an 8 kHz sample rate and FFT size 128,
    giving a frequency-bin spacing of 62.5 Hz.

    Default band bin counts:
        0–500 Hz:
            8 bins with the half-open interval convention

        500–2000 Hz:
            24 bins

        2000–4000 Hz:
            33 bins because the final band includes Nyquist

    Every band therefore exceeds the default minimum of four bins.
    """
    config = SpectralFlatnessLossConfig(
        enabled=True,
        fft_size=fft_size,
        hop_size=hop_size,
        win_length=win_length,
        band_edges=band_edges,
        energy_floor=energy_floor,
        minimum_bins_per_band=(
            minimum_bins_per_band
        ),
        eps=eps,
    )

    loss_function = (
        SubBandSpectralFlatnessLoss(
            sample_rate=sample_rate,
            config=config,
        )
    )

    return loss_function.to(
        torch.device(device)
    )


def make_test_dc_offset_output(
    *,
    device: torch.device | str = "cpu",
) -> MultiScaleDCOffsetLossOutput:
    scales = (
        DCOffsetScaleLossOutput(
            window_size=16,
            hop_size=4,
            loss=scalar_tensor(
                0.1,
                device=device,
            ),
        ),
        DCOffsetScaleLossOutput(
            window_size=64,
            hop_size=16,
            loss=scalar_tensor(
                0.2,
                device=device,
            ),
        ),
    )
    local = torch.stack(
        [
            scale.loss
            for scale in scales
        ]
    ).mean()

    global_mean = scalar_tensor(
        0.05,
        device=device,
    )

    total = 0.5 * (
        local + global_mean
    )
    return MultiScaleDCOffsetLossOutput(
        total=total,
        local=local,
        global_mean=global_mean,
        scales=scales,
    )


def make_test_cqt_analysis_config(
    *,
    sample_rate: int = 8_000,
    hop_length: int = 32,
    fmin: float = 62.5,
    number_of_bins: int = 36,
    bins_per_octave: int = 12,
    filter_scale: float = 1.0,
    norm: int = 1,
) -> CQTAnalysisConfig:
    return CQTAnalysisConfig(
        sample_rate=sample_rate,
        hop_length=hop_length,
        fmin=fmin,
        number_of_bins=number_of_bins,
        bins_per_octave=bins_per_octave,
        filter_scale=filter_scale,
        norm=norm,
        window="hann",
        center=True,
        pad_mode="reflect",
    )


def make_test_cqt_loss_config(
    *,
    enabled: bool = True,
    eps: float = 1e-7,
) -> CQTConsistencyLossConfig:
    return CQTConsistencyLossConfig(
        enabled=enabled,
        eps=eps,
        distance="log_magnitude_l1",
        analysis=make_test_cqt_analysis_config(),
    )


def make_test_cqt_loss(
    *,
    device: torch.device | str = "cpu",
    **analysis_overrides,
) -> CQTLogMagnitudeLoss:

    loss_config = (
        make_test_cqt_loss_config()
    )
    return CQTLogMagnitudeLoss(
        loss_config=loss_config,
    ).to(
        torch.device(device)
    )


def make_test_rail_occupancy_config(
    *,
    enabled: bool = True,
    window_sizes: tuple[int, ...] = (
        16,
        64,
    ),
    hop_ratio: int = 4,
    threshold: float = 0.8,
    sharpness: float = 20.0,
    activity_floor: float = 1e-2,
    eps: float = 1e-8,
) -> RailOccupancyLossConfig:
    return RailOccupancyLossConfig(
        enabled=enabled,
        window_sizes=window_sizes,
        hop_ratio=hop_ratio,
        threshold=threshold,
        sharpness=sharpness,
        activity_floor=activity_floor,
        eps=eps,
    )


def make_test_rail_occupancy_loss(
    **config_overrides,
) -> MultiScaleRailOccupancyLoss:
    return MultiScaleRailOccupancyLoss(
        make_test_rail_occupancy_config(
            **config_overrides
        )
    )


def make_test_rail_occupancy_output(
    *,
    device: torch.device | str = "cpu",
) -> MultiScaleRailOccupancyLossOutput:
    scales = (
        RailOccupancyScaleLossOutput(
            window_size=16,
            hop_size=4,
            loss=scalar_tensor(
                0.1,
                device=device,
            ),
            target_occupancy=scalar_tensor(
                0.8,
                device=device,
            ),
            prediction_occupancy=scalar_tensor(
                0.7,
                device=device,
            ),
            target_peak_mean=scalar_tensor(
                0.9,
                device=device,
            ),
            activity_fraction=scalar_tensor(
                0.95,
                device=device,
            ),
        ),
        RailOccupancyScaleLossOutput(
            window_size=64,
            hop_size=16,
            loss=scalar_tensor(
                0.2,
                device=device,
            ),
            target_occupancy=scalar_tensor(
                0.75,
                device=device,
            ),
            prediction_occupancy=scalar_tensor(
                0.6,
                device=device,
            ),
            target_peak_mean=scalar_tensor(
                0.85,
                device=device,
            ),
            activity_fraction=scalar_tensor(
                0.9,
                device=device,
            ),
        ),
    )

    total = torch.stack(
        [
            scale.loss
            for scale in scales
        ]
    ).mean()

    target_occupancy = torch.stack(
        [
            scale.target_occupancy
            for scale in scales
        ]
    ).mean()

    prediction_occupancy = torch.stack(
        [
            scale.prediction_occupancy
            for scale in scales
        ]
    ).mean()

    return MultiScaleRailOccupancyLossOutput(
        total=total,
        target_occupancy=target_occupancy,
        prediction_occupancy=(
            prediction_occupancy
        ),
        scales=scales,
    )


def make_test_weighted_component(
    *,
    raw: float,
    weight: float,
    device: torch.device | str = "cpu",
) -> WeightedLossComponent:
    raw_tensor = scalar_tensor(
        raw,
        device=device,
    )

    return make_weighted_loss_component(
        raw=raw_tensor,
        weight=weight,
    )


def make_test_generator_loss_output(
    *,
    adversarial_raw: float = 2.0,
    adversarial_weight: float = 0.5,
    feature_matching_raw: float = 3.0,
    feature_matching_weight: float = 2.0,
    mrstft_raw: float = 4.0,
    mrstft_weight: float = 1.5,
    waveform_raw: float = 5.0,
    waveform_weight: float = 0.2,
    temporal_gradient_raw: float = 6.0,
    temporal_gradient_weight: float = 0.1,
    temporal_gradient_output: MultiScaleSlewEnvelopeLossOutput | None = None,
    rms_envelope_raw: float = 7.0,
    rms_envelope_weight: float = 0.05,
    rms_envelope_output: MultiScaleRMSEnvelopeLossOutput | None = None,
    band_power_raw: float = 2.0,
    band_power_weight: float = 1.0,
    band_power_output: MultiResolutionBandPowerLossOutput | None = None,
    adversarial_families: (
        dict[str, torch.Tensor]
        | None
    ) = None,
    feature_matching_families: (
            dict[str, torch.Tensor]
        | None
    ) = None,
    mrstft_output: (
        MultiResolutionSTFTLossOutput
        | None
    ) = None,
    spectral_flatness_raw: float = 0.5,
    spectral_flatness_weight: float = 0.1,
    spectral_flatness_output: (
        SubBandSpectralFlatnessLossOutput
        | None
    ) = None,
    dc_offset_raw: float = 0.05,
    dc_offset_weight: float = 0.01,
    dc_offset_output: (
        MultiScaleDCOffsetLossOutput
        | None
    ) = None,
    cqt_consistency_raw: float = 2.0,
    cqt_consistency_weight: float = 1.0,
    cqt_consistency_output: CQTConsistencyLossOutput | None = None,
    rail_occupancy_raw: float = 2.0,
    rail_occupancy_weight: float = 1.0,
    rail_occupancy_output: MultiScaleRailOccupancyLossOutput | None = None,
    device: torch.device | str = "cpu",
) -> VocoderGeneratorLossOutput:
    adversarial = make_test_weighted_component(
        raw=adversarial_raw,
        weight=adversarial_weight,
        device=device,
    )

    feature_matching = make_test_weighted_component(
        raw=feature_matching_raw,
        weight=feature_matching_weight,
        device=device,
    )

    mrstft = make_test_weighted_component(
        raw=mrstft_raw,
        weight=mrstft_weight,
        device=device,
    )

    waveform = make_test_weighted_component(
        raw=waveform_raw,
        weight=waveform_weight,
        device=device,
    )

    temporal_gradient = make_test_weighted_component(
        raw=temporal_gradient_raw,
        weight=temporal_gradient_weight,
        device=device,
    )

    rms_envelope = make_test_weighted_component(
        raw=rms_envelope_raw,
        weight=rms_envelope_weight,
        device=device,
    )

    band_power = make_test_weighted_component(
        raw=band_power_raw,
        weight=band_power_weight,
        device=device,
    )

    spectral_flatness = (
        make_test_weighted_component(
            raw=spectral_flatness_raw,
            weight=spectral_flatness_weight,
            device=device,
        )
    )

    dc_offset = make_test_weighted_component(
        raw=dc_offset_raw,
        weight=dc_offset_weight,
        device=device,
    )

    cqt_consistency = make_test_weighted_component(
        raw=cqt_consistency_raw,
        weight=cqt_consistency_weight,
        device=device,
    )

    rail_occupancy = make_test_weighted_component(
        raw=rail_occupancy_raw,
        weight=rail_occupancy_weight,
        device=device,
    )

    if adversarial_families is None:
        adversarial_families = {
            "mpd": scalar_tensor(
                adversarial_raw,
                device=device,
            )
        }

    if feature_matching_families is None:
        feature_matching_families = {
            "mpd": scalar_tensor(
                feature_matching_raw,
                device=device,
            )
        }

    if mrstft_output is None:
        mrstft_output = make_test_mrstft_output(
            device=device,
        )

    total = (
        adversarial.weighted
        + feature_matching.weighted
        + mrstft.weighted
        + waveform.weighted
        + temporal_gradient.weighted
        + rms_envelope.weighted
        + band_power.weighted
        + spectral_flatness.weighted
        + dc_offset.weighted
        + cqt_consistency.weighted
        + rail_occupancy.weighted
    )

    return VocoderGeneratorLossOutput(
        total=total,
        adversarial=adversarial,
        feature_matching=(
            feature_matching
        ),
        mrstft=mrstft,
        waveform=waveform,
        temporal_gradient=(
            temporal_gradient
        ),
        rms_envelope=rms_envelope,
        band_power=band_power,
        spectral_flatness=spectral_flatness,
        dc_offset=dc_offset,
        cqt_consistency=cqt_consistency,
        rail_occupancy=rail_occupancy,
        adversarial_families=(
            adversarial_families
        ),
        feature_matching_families=(
            feature_matching_families
        ),
        mrstft_output=mrstft_output,
        temporal_gradient_output=temporal_gradient_output,
        rms_envelope_output=rms_envelope_output,
        band_power_output=band_power_output,
        spectral_flatness_output=spectral_flatness_output,
        dc_offset_output=dc_offset_output,
        cqt_consistency_output=cqt_consistency_output,
        rail_occupancy_output=rail_occupancy_output,
    )



def make_zero_generator_loss_output(
    *,
    device: torch.device | str = "cpu",
) -> VocoderGeneratorLossOutput:
    return make_test_generator_loss_output(
        adversarial_raw=0.0,
        adversarial_weight=1.0,
        feature_matching_raw=0.0,
        feature_matching_weight=1.0,
        mrstft_raw=0.0,
        mrstft_weight=1.0,
        waveform_raw=0.0,
        waveform_weight=1.0,
        temporal_gradient_raw=0.0,
        temporal_gradient_weight=1.0,
        mrstft_output=make_zero_mrstft_output(
            device=device,
        ),
        device=device,
    )


def make_test_loss_config(
    *,
    lambda_adversarial: float = 0.5,
    lambda_feature_matching: float = 2.0,
    lambda_mrstft: float = 1.5,
    lambda_waveform: float = 0.2,
    lambda_temporal_gradient: float = 0.1,
    feature_matching_normalization: str = "layer_mean",
    temporal_gradient_enabled: bool = True,
) -> VocoderLossConfig:
    return VocoderLossConfig(
        lambda_adversarial=(
            lambda_adversarial
        ),
        lambda_feature_matching=(
            lambda_feature_matching
        ),
        lambda_mrstft=lambda_mrstft,
        lambda_waveform=lambda_waveform,
        lambda_temporal_gradient=(
            lambda_temporal_gradient
        ),
        feature_matching=FeatureMatchingLossConfig(
            normalization=(
                feature_matching_normalization
            ),
            scale_floor=1e-2,
            eps=1e-8,
        ),
        mrstft=MRSTFTConfig(
            fft_sizes=(128, 256),
            hop_sizes=(32, 64),
            win_lengths=(128, 256),
            spectral_convergence_weight=1.0,
            log_magnitude_weight=1.0,
            eps=1e-7,
        ),
        temporal_gradient=TemporalGradientLossConfig(
            enabled=temporal_gradient_enabled,
            lags=(1, 2, 4),
            normalization="none",
            scale_floor=1e-3,
            eps=1e-8,
            edge_weighting=False,
            edge_weight=1.0,
            max_edge_weight=5.0,
        ),
    )


def make_test_discriminator_outputs(
    *,
    batch_size: int = 2,
    device: torch.device | str = "cpu",
    fake_requires_grad: bool = True,
) -> dict[str, list]:
    real_output = torch.full(
        (batch_size, 4),
        0.8,
        dtype=torch.float32,
        device=device,
    )

    fake_output = torch.full(
        (batch_size, 4),
        0.2,
        dtype=torch.float32,
        device=device,
        requires_grad=fake_requires_grad,
    )

    real_feature_map = torch.full(
        (batch_size, 4, 8),
        0.5,
        dtype=torch.float32,
        device=device,
    )

    fake_feature_map = torch.full(
        (batch_size, 4, 8),
        0.3,
        dtype=torch.float32,
        device=device,
        requires_grad=fake_requires_grad,
    )

    return {
        "real_outputs": [
            real_output,
        ],
        "fake_outputs": [
            fake_output,
        ],
        "real_feature_maps": [
            [
                real_feature_map,
            ]
        ],
        "fake_feature_maps": [
            [
                fake_feature_map,
            ]
        ],

        "mpd_real_outputs": [
            real_output,
        ],
        "mpd_fake_outputs": [
            fake_output,
        ],
        "mpd_real_feature_maps": [
            [
                real_feature_map,
            ]
        ],
        "mpd_fake_feature_maps": [
            [
                fake_feature_map,
            ]
        ],

        "msd_real_outputs": [],
        "msd_fake_outputs": [],
        "msd_real_feature_maps": [],
        "msd_fake_feature_maps": [],

        "mrd_real_outputs": [],
        "mrd_fake_outputs": [],
        "mrd_real_feature_maps": [],
        "mrd_fake_feature_maps": [],
    }


def make_family_outputs(
    values: tuple[float, ...],
    *,
    batch_size: int = 2,
    patch_count: int = 4,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> list[torch.Tensor]:
    return [
        torch.full(
            (batch_size, patch_count),
            value,
            dtype=torch.float32,
            device=device,
            requires_grad=requires_grad,
        )
        for value in values
    ]


def make_test_waveform_pair(
    *,
    batch_size: int = 2,
    channels: int = 1,
    length: int = 2048,
    device: torch.device | str = "cpu",
    fake_requires_grad: bool = True,
    seed: int = 1337,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(
        device=device,
    )
    generator.manual_seed(seed)

    real = torch.randn(
        batch_size,
        channels,
        length,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )

    fake = torch.randn(
        batch_size,
        channels,
        length,
        dtype=torch.float32,
        device=device,
        generator=generator,
        requires_grad=fake_requires_grad,
    )

    return real, fake


def assert_scalar_tensor(
    value: torch.Tensor,
    *,
    finite: bool = True,
) -> None:
    assert isinstance(value, torch.Tensor)
    assert value.ndim == 0

    if finite:
        assert torch.isfinite(value)


def assert_weighted_component_consistent(
    component: WeightedLossComponent,
) -> None:
    assert_scalar_tensor(component.raw)
    assert_scalar_tensor(component.weighted)

    expected = (
        component.raw
        * component.weight
    )

    assert torch.allclose(
        component.weighted,
        expected,
        atol=1e-7,
        rtol=1e-6,
    )


def assert_generator_loss_output_consistent(
    output: VocoderGeneratorLossOutput,
) -> None:
    assert_scalar_tensor(output.total)

    components = (
        output.adversarial,
        output.feature_matching,
        output.mrstft,
        output.waveform,
        output.temporal_gradient,
    )

    for component in components:
        assert_weighted_component_consistent(
            component
        )

    expected_total = torch.stack(
        [
            component.weighted
            for component in components
        ]
    ).sum()

    assert torch.allclose(
        output.total,
        expected_total,
        atol=1e-7,
        rtol=1e-6,
    )

    assert output.adversarial_families
    assert output.feature_matching_families

    expected_adversarial = torch.stack(
        tuple(
            output.adversarial_families.values()
        )
    ).mean()

    expected_feature_matching = torch.stack(
        tuple(
            output
            .feature_matching_families
            .values()
        )
    ).mean()

    assert torch.allclose(
        output.adversarial.raw,
        expected_adversarial,
        atol=1e-7,
        rtol=1e-6,
    )

    assert torch.allclose(
        output.feature_matching.raw,
        expected_feature_matching,
        atol=1e-7,
        rtol=1e-6,
    )

    assert torch.allclose(
        output.learned_total,
        (
            output.adversarial.weighted
            + output.feature_matching.weighted
        ),
    )

    assert torch.allclose(
        output.time_micro_total,
        (
            output.waveform.weighted
            + output.temporal_gradient.weighted
        ),
    )

    assert torch.allclose(
        output.time_frequency_total,
        output.mrstft.weighted,
    )


def assert_finite_metrics(
    metrics: Mapping[str, float],
) -> None:
    assert metrics

    for name, value in metrics.items():
        assert isinstance(value, float), (
            f"Metric {name!r} is not a float: "
            f"{type(value).__name__}"
        )

        assert torch.isfinite(
            torch.tensor(value)
        ), (
            f"Metric {name!r} is non-finite: {value}"
        )


def make_family_prediction_outputs(
    values: tuple[float, ...],
    *,
    batch_size: int = 1,
    patch_count: int = 4,
    requires_grad: bool = False,
    device: torch.device | str = "cpu",
) -> list[torch.Tensor]:
    return [
        torch.full(
            (batch_size, patch_count),
            fill_value=value,
            dtype=torch.float32,
            device=device,
            requires_grad=requires_grad,
        )
        for value in values
    ]


def make_family_feature_maps(
    branch_values: tuple[
        tuple[float, ...],
        ...,
    ],
    *,
    batch_size: int = 1,
    channels: int = 2,
    time: int = 4,
    requires_grad: bool = False,
    device: torch.device | str = "cpu",
) -> list[list[torch.Tensor]]:
    """
    branch_values describes one tuple per discriminator branch.

    Example:
        (
            (1.0, 2.0),  # Branch 0, two layers
            (3.0,),      # Branch 1, one layer
        )
    """
    return [
        [
            torch.full(
                (
                    batch_size,
                    channels,
                    time,
                ),
                fill_value=layer_value,
                dtype=torch.float32,
                device=device,
                requires_grad=requires_grad,
            )
            for layer_value in layer_values
        ]
        for layer_values in branch_values
    ]


def make_empty_discriminator_family() -> dict[str, list]:
    return {
        "real_outputs": [],
        "fake_outputs": [],
        "real_feature_maps": [],
        "fake_feature_maps": [],
    }


def make_discriminator_family(
    *,
    real_output_values: tuple[float, ...],
    fake_output_values: tuple[float, ...],
    real_feature_values: (
        tuple[tuple[float, ...], ...]
        | None
    ) = None,
    fake_feature_values: (
            tuple[tuple[float, ...], ...]
        | None
    ) = None,
    batch_size: int = 1,
    patch_count: int = 4,
    feature_channels: int = 2,
    feature_time: int = 4,
    fake_requires_grad: bool = True,
    device: torch.device | str = "cpu",
) -> dict[str, list]:
    if len(real_output_values) != len(
        fake_output_values
    ):
        raise ValueError(
            "Real and fake output counts must match."
        )

    if real_feature_values is None:
        real_feature_values = [(value, ) for value in real_output_values]

    if fake_feature_values is None:
        fake_feature_values = [(value, ) for value in fake_output_values]

    if len(real_feature_values) != len(real_output_values):
        raise ValueError(
            "Real feature-map branch count must match "
            "the output branch count."
        )

    if len(fake_feature_values) != len(fake_output_values):
        raise ValueError(
            "Fake feature-map branch count must match "
            "the output branch count."
        )

    return {
        "real_outputs": (
            make_family_prediction_outputs(
                real_output_values,
                batch_size=batch_size,
                patch_count=patch_count,
                requires_grad=False,
                device=device,
            )
        ),
        "fake_outputs": (
            make_family_prediction_outputs(
                fake_output_values,
                batch_size=batch_size,
                patch_count=patch_count,
                requires_grad=fake_requires_grad,
                device=device,
            )
        ),
        "real_feature_maps": (
            make_family_feature_maps(
                real_feature_values,
                batch_size=batch_size,
                channels=feature_channels,
                time=feature_time,
                requires_grad=False,
                device=device,
            )
        ),
        "fake_feature_maps": (
            make_family_feature_maps(
                fake_feature_values,
                batch_size=batch_size,
                channels=feature_channels,
                time=feature_time,
                requires_grad=fake_requires_grad,
                device=device,
            )
        ),
    }


def make_multi_family_discriminator_outputs(
    *,
    mpd: dict[str, list] | None = None,
    msd: dict[str, list] | None = None,
    mrd: dict[str, list] | None = None,
) -> dict[str, list]:
    if mpd is None:
        mpd = make_empty_discriminator_family()

    if msd is None:
        msd = make_empty_discriminator_family()

    if mrd is None:
        mrd = make_empty_discriminator_family()

    result: dict[str, list] = {}

    for family_name, family in (
        ("mpd", mpd),
        ("msd", msd),
        ("mrd", mrd),
    ):
        result[
            f"{family_name}_real_outputs"
        ] = family["real_outputs"]

        result[
            f"{family_name}_fake_outputs"
        ] = family["fake_outputs"]
        result[
            f"{family_name}_real_feature_maps"
        ] = family["real_feature_maps"]

        result[
            f"{family_name}_fake_feature_maps"
        ] = family["fake_feature_maps"]

    result["real_outputs"] = (
        mpd["real_outputs"]
        + msd["real_outputs"]
        + mrd["real_outputs"]
    )

    result["fake_outputs"] = (
        mpd["fake_outputs"]
        + msd["fake_outputs"]
        + mrd["fake_outputs"]
    )

    result["real_feature_maps"] = (
        mpd["real_feature_maps"]
        + msd["real_feature_maps"]
        + mrd["real_feature_maps"]
    )

    result["fake_feature_maps"] = (
        mpd["fake_feature_maps"]
        + msd["fake_feature_maps"]
        + mrd["fake_feature_maps"]
    )

    return result


def assert_family_loss_output_consistent(
     output: FamilyLossOutput,
) -> None:

    assert_scalar_tensor(output.total)

    assert output.families

    for family_name, value in (output.families.items()):
        assert family_name in {
            "mpd",
            "msd",
            "mrd",
        }

        assert_scalar_tensor(value)

    expected_total = torch.stack(tuple(output.families.values())).mean()

    assert torch.allclose(output.total, expected_total, atol=1e-7, rtol = 1e-6)


def assert_family_discriminator_loss_consistent(
    output: FamilyDiscriminatorLossOutput,
) -> None:
    assert_scalar_tensor(output.total)
    assert_scalar_tensor(output.real)
    assert_scalar_tensor(output.fake)

    assert output.family_totals
    assert output.family_real
    assert output.family_fake

    assert set(output.family_totals) == set(output.family_real)

    assert set(output.family_totals) == set(output.family_fake)

    for family in output.family_totals:
        assert_scalar_tensor(
            output.family_totals[family]
        )
        assert_scalar_tensor(
            output.family_real[family]
        )
        assert_scalar_tensor(
            output.family_fake[family]
        )

        assert torch.allclose(
            output.family_totals[family],
            (
                output.family_real[family]
                + output.family_fake[family]
            ),
            atol=1e-7,
            rtol=1e-6,
        )
    expected_total = torch.stack(tuple(output.family_totals.values())).mean()

    expected_real = torch.stack(tuple(output.family_real.values())).mean()

    expected_fake = torch.stack(tuple(output.family_fake.values())).mean()

    assert torch.allclose(
        output.total,
        expected_total,
        atol=1e-7,
        rtol=1e-6,
    )

    assert torch.allclose(
        output.real,
        expected_real,
        atol=1e-7,
        rtol=1e-6,
    )

    assert torch.allclose(
        output.fake,
        expected_fake,
        atol=1e-7,
        rtol=1e-6,
    )

    assert torch.allclose(
        output.total,
        output.real + output.fake,
        atol=1e-7,
        rtol=1e-6,
    )


