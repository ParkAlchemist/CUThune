from __future__ import annotations

from vocoder.src.nes_vocoder.config.schema import VocoderExperimentConfig, VocoderGeneratorModelConfig, MRSTFTConfig, \
    VocoderLossConfig, TemporalGradientLossConfig, FeatureMatchingLossConfig, RMSEnvelopeLossConfig, \
    BandPowerLossConfig, SpectralFlatnessLossConfig, DCOffsetLossConfig, CQTConsistencyLossConfig, CQTAnalysisConfig, \
    RailOccupancyLossConfig

from src.models.vocoder_discriminators import (
    MultiPeriodDiscriminatorConfig as ModelMPDConfig,
    MultiScaleDiscriminatorConfig as ModelMSDConfig,
    ScaleDiscriminatorConfig as ModelScaleConfig,
    VocoderDiscriminatorConfig as ModelDiscriminatorConfig,
)



def build_generator_config(
    experiment: VocoderExperimentConfig,
) -> VocoderGeneratorModelConfig:
    source = experiment.generator

    return VocoderGeneratorModelConfig(
        upsample_initial_channel=source.upsample_initial_channel,
        upsample_rates=source.upsample_rates,
        upsample_kernel_sizes=source.upsample_kernel_sizes,
        resblock_kernel_sizes=source.resblock_kernel_sizes,
        resblock_dilation_sizes=source.resblock_dilation_sizes,
        activation=source.activation,
        final_tanh=source.final_tanh,
        use_weight_norm=source.use_weight_norm,
    )


def build_mrstft_config(
    experiment: VocoderExperimentConfig,
) -> MRSTFTConfig:
    source = experiment.loss.mrstft

    return MRSTFTConfig(
        fft_sizes=source.fft_sizes,
        hop_sizes=source.hop_sizes,
        win_lengths=source.win_lengths,
        spectral_convergence_weight=(
            source.spectral_convergence_weight
        ),
        log_magnitude_weight=source.log_magnitude_weight,
        eps=source.eps,
        reference_floor=source.reference_floor,
    )


def build_temporal_gradient_config(
        experiment: VocoderExperimentConfig,
) -> TemporalGradientLossConfig:
    source = experiment.loss.temporal_gradient

    return TemporalGradientLossConfig(
        enabled=source.enabled,
        mode=source.mode,
        lags=source.lags,
        normalization=source.normalization,
        scale_floor=source.scale_floor,
        eps=source.eps,
        edge_weighting=source.edge_weighting,
        edge_weight=source.edge_weight,
        max_edge_weight=source.max_edge_weight,
        window_sizes=source.window_sizes,
        hop_ratio=source.hop_ratio,
        mean_weight=source.mean_weight,
        peak_weight=source.peak_weight,
    )


def build_feature_matching_config(
    experiment: VocoderExperimentConfig,
) -> FeatureMatchingLossConfig:
    source = experiment.loss.feature_matching

    return FeatureMatchingLossConfig(
        normalization=source.normalization,
        scale_floor=source.scale_floor,
        eps=source.eps,
    )


def build_rms_envelope_loss_config(
    experiment: VocoderExperimentConfig,
) -> RMSEnvelopeLossConfig:
    source = experiment.loss.rms_envelope

    return RMSEnvelopeLossConfig(
        enabled=source.enabled,
        window_sizes=source.window_sizes,
        hop_ratio=source.hop_ratio,
        eps=source.eps,
    )


def build_band_power_loss_config(
        experiment: VocoderExperimentConfig,
) -> BandPowerLossConfig:
    source = experiment.loss.band_power

    return BandPowerLossConfig(
        enabled=source.enabled,
        fft_sizes=source.fft_sizes,
        hop_sizes=source.hop_sizes,
        win_lengths=source.win_lengths,
        number_of_bands=source.number_of_bands,
        fmin=source.fmin,
        fmax=source.fmax,
        absolute_weight=source.absolute_weight,
        shape_weight=source.shape_weight,
        eps=source.eps,
        energy_floor=source.energy_floor,
    )


def build_spectral_flatness_loss_config(
    experiment: VocoderExperimentConfig,
) -> SpectralFlatnessLossConfig:
    source = experiment.loss.spectral_flatness

    return SpectralFlatnessLossConfig(
        enabled=source.enabled,
        fft_size=source.fft_size,
        hop_size=source.hop_size,
        win_length=source.win_length,
        band_edges=tuple(source.band_edges),
        energy_floor=source.energy_floor,
        minimum_bins_per_band=(
            source.minimum_bins_per_band
        ),
        eps=source.eps,
    )


def build_dc_offset_loss_config(
    experiment: VocoderExperimentConfig,
) -> DCOffsetLossConfig:
    source = experiment.loss.dc_offset

    return DCOffsetLossConfig(
        enabled=source.enabled,
        window_sizes=tuple(
            source.window_sizes
        ),
        hop_ratio=source.hop_ratio,
    )


def build_cqt_consistency_loss_config(
    experiment: VocoderExperimentConfig,
) -> CQTConsistencyLossConfig:
    source = experiment.loss.cqt
    data = experiment.data

    return CQTConsistencyLossConfig(
        enabled=source.enabled,
        eps=source.eps,
        distance=source.distance,
        analysis=CQTAnalysisConfig(
            sample_rate=data.sample_rate,
            hop_length=data.hop_length,
            fmin=data.fmin,
            number_of_bins=data.cqt_bins,
            bins_per_octave=data.bins_per_octave,
        )
    )


def build_rail_occupancy_loss_config(
    experiment: VocoderExperimentConfig,
) -> RailOccupancyLossConfig:
    source = experiment.loss.rail_occupancy

    return RailOccupancyLossConfig(
        enabled=source.enabled,
        window_sizes=tuple(
            source.window_sizes
        ),
        hop_ratio=source.hop_ratio,
        threshold=source.threshold,
        sharpness=source.sharpness,
        activity_floor=source.activity_floor,
        eps=source.eps,
    )


def build_loss_config(
    experiment: VocoderExperimentConfig,
) -> VocoderLossConfig:
    source = experiment.loss

    return VocoderLossConfig(
        lambda_adversarial=source.lambda_adversarial,
        lambda_feature_matching=source.lambda_feature_matching,
        lambda_mrstft=source.lambda_mrstft,
        lambda_waveform=source.lambda_waveform,
        lambda_temporal_gradient=source.lambda_temporal_gradient,
        lambda_rms_envelope=source.lambda_rms_envelope,
        lambda_band_power=source.lambda_band_power,
        lambda_spectral_flatness=source.lambda_spectral_flatness,
        lambda_dc_offset=source.lambda_dc_offset,
        lambda_cqt=source.lambda_cqt,
        lambda_rail_occupancy=source.lambda_rail_occupancy,
        mrstft=build_mrstft_config(experiment),
        temporal_gradient=build_temporal_gradient_config(experiment),
        feature_matching=build_feature_matching_config(experiment),
        rms_envelope=build_rms_envelope_loss_config(experiment),
        band_power=build_band_power_loss_config(experiment),
        spectral_flatness=build_spectral_flatness_loss_config(experiment),
        dc_offset=build_dc_offset_loss_config(experiment),
        cqt=build_cqt_consistency_loss_config(experiment),
        rail_occupancy=build_rail_occupancy_loss_config(experiment),
    )


def build_discriminator_config(
    experiment: VocoderExperimentConfig,
) -> ModelDiscriminatorConfig:
    source = experiment.discriminator

    mpd = ModelMPDConfig(
        periods=source.mpd.periods,
        channels=source.mpd.channels,
        kernel_size=source.mpd.kernel_size,
        stride=source.mpd.stride,
        norm=source.mpd.norm,
        negative_slope=source.mpd.negative_slope,
    )

    scale = ModelScaleConfig(
        channels=source.msd.discriminator.channels,
        kernel_sizes=source.msd.discriminator.kernel_sizes,
        strides=source.msd.discriminator.strides,
        groups=source.msd.discriminator.groups,
        norm=source.msd.discriminator.norm,
        negative_slope=source.msd.discriminator.negative_slope,
    )

    msd = ModelMSDConfig(
        num_scales=source.msd.num_scales,
        first_discriminator_norm=(
            source.msd.first_discriminator_norm
        ),
        other_discriminator_norm=(
            source.msd.other_discriminator_norm
        ),
        pool_kernel_size=source.msd.pool_kernel_size,
        pool_stride=source.msd.pool_stride,
        pool_padding=source.msd.pool_padding,
        discriminator=scale,
    )

    return ModelDiscriminatorConfig(
        mpd=mpd,
        msd=msd,
        use_mpd=source.use_mpd,
        use_msd=source.use_msd,
    )

