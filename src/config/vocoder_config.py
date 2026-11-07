from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


NormType = Literal["weight",  "spectral", "none"]
ActivationType = Literal["leaky_relu", "snake_beta"]
SchedulerName = Literal["none", "exponential", "cosine"]
SchedulerInterval = Literal["epoch", "optimizer_step"]
ContextBlockType = Literal["convnext", "multi_kernel_convnext"]
FeatureMatchingNormalization = Literal["layer_mean", "real_magnitude", "real_rms"]
TemporalLossMode = Literal["pointwise_gradient", "slew_envelope"]


@dataclass
class RunConfig:
    experiment_name: str = "vocoder"
    output_root: str = "runs/vocoder_cqt"
    seed: int = 1337
    device: str = "cuda"


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 2e-4
    weight_decay: float = 0.0
    beta1: float = 0.8
    beta2: float = 0.99
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    name: SchedulerName = "exponential"
    enabled: bool = False
    gamma: float = 0.999
    interval: SchedulerInterval = "epoch"
    t_max: int = 500
    eta_min: float = 0.0


@dataclass
class TrainingConfig:
    epochs: int = 500
    max_steps: int | None = None
    grad_clip_generator: float = 10.0
    grad_clip_discriminator: float = 10.0

    gradient_accumulation_steps: int = 1
    generator_start_step: int = 0
    discriminator_start_step: int = 0
    adversarial_start_step: int = 0

    overfit_batches: int = 0
    detect_anomaly: bool = False
    fail_on_nonfinite: bool = True


@dataclass
class AMPConfig:
    enabled: bool = False
    start_step: int = 0
    dtype: str = "float16"
    initial_scale: float = 256.0
    growth_interval: int = 2000


@dataclass
class AugmentationConfig:
    enabled: bool = False
    gain_db_min: float = 0.0
    gain_db_max: float = 0.0
    conditioning_noise_std: float = 0.0
    time_mask_probability: float = 0.0
    frequency_mask_probability: float = 0.0


@dataclass
class LogConfig:
    log_every_steps: int = 50
    status_every_steps: int = 50
    save_every_steps: int = 10000
    preview_every_steps: int = 10000
    preview_num_samples: int = 4
    keep_numbered_checkpoints: int = 10
    tensorboard: bool = True


@dataclass
class ControlConfig:
    enabled: bool = True
    poll_every_steps: int = 10
    save_on_interrupt: bool = True


@dataclass
class MRSTFTConfig:
    fft_sizes: tuple[int, ...] = (128, 256, 512, 1024, 2048)
    hop_sizes: tuple[int, ...] = (32, 64, 128, 256, 512)
    win_lengths: tuple[int, ...] = (128, 256, 512, 1024, 2048)
    spectral_convergence_weight: float = 1.0
    log_magnitude_weight: float = 1.0
    eps: float = 1e-7
    reference_floor: float = 1.0


@dataclass
class FeatureMatchingLossConfig:
    normalization: FeatureMatchingNormalization = "layer_mean"
    scale_floor: float = 1e-2
    eps: float = 1e-8


@dataclass
class TemporalGradientLossConfig:
    enabled: bool = False

    mode: TemporalLossMode = "pointwise_gradient"

    lags: tuple[int, ...] = (1, 2, 4)
    normalization: Literal["none", "target_magnitude"] = "none"
    scale_floor: float = 1e-3
    eps: float = 1e-8

    edge_weighting: bool = False
    edge_weight: float = 1.0
    max_edge_weight: float = 5.0

    window_sizes: tuple[int, ...] = (
        32,
        128,
        256,
    )

    hop_ratio: int = 4

    mean_weight: float = 1.0
    peak_weight: float = 0.5


@dataclass
class RMSEnvelopeLossConfig:
    enabled: bool = False

    window_sizes: tuple[int, ...] = (
        64,
        256,
        1024,
        4096,
    )

    hop_ratio: int = 4
    eps: float = 1e-8


@dataclass
class BandPowerLossConfig:
    enabled: bool = False

    fft_sizes: tuple[int, ...] = (
        1024,
        4096,
    )

    hop_sizes: tuple[int, ...] = (
        256,
        1024,
    )

    win_lengths: tuple[int, ...] = (
        1024,
        4096,
    )

    number_of_bands: int = 24

    fmin: float = 27.5
    fmax: float | None = None

    absolute_weight: float = 1.0
    shape_weight: float = 1.0

    eps: float = 1e-8
    energy_floor: float = 1e-4


@dataclass
class SpectralFlatnessLossConfig:
    enabled: bool = False

    fft_size: int = 1024
    hop_size: int = 256
    win_length: int = 1024

    band_edges: tuple[float, ...] = (
        27.5,
        250.0,
        1000.0,
        4000.0,
        11025.0,
    )

    energy_floor: float = 1e-5
    minimum_bins_per_band: int = 4
    eps: float = 1e-8


@dataclass
class DCOffsetLossConfig:
    enabled: bool = False

    window_sizes: tuple[int, ...] = (
        1024,
        4096,
        16384,
    )

    hop_ratio: int = 4

@dataclass(frozen=True)
class CQTAnalysisConfig:
    sample_rate: int = 8000
    hop_length: int = 32
    fmin: float = 62.5
    number_of_bins: int = 36
    bins_per_octave: int = 12

    filter_scale: float = 1.0
    norm: int = 1
    window: str = "hann"
    center: bool = True
    pad_mode: str = "reflect"


@dataclass
class CQTConsistencyLossConfig:
    enabled: bool = False
    eps: float = 1e-7
    distance: Literal["log_magnitude_l1"] = "log_magnitude_l1"
    analysis: CQTAnalysisConfig = field(default_factory=CQTAnalysisConfig)


@dataclass
class RailOccupancyLossConfig:
    enabled: bool = False

    window_sizes: tuple[int, ...] = (
        32,
        128,
        256,
    )

    hop_ratio: int = 4

    threshold: float = 0.8
    sharpness: float = 20.0

    activity_floor: float = 1e-2
    eps: float = 1e-8


@dataclass
class VocoderLossConfig:
    lambda_adversarial: float = 1.0
    lambda_feature_matching: float = 2.0
    lambda_mrstft: float = 1.25
    lambda_waveform: float = 0.5
    lambda_temporal_gradient: float = 0.1
    lambda_rms_envelope: float = 0.0
    lambda_band_power: float = 0.0
    lambda_spectral_flatness: float = 0.0
    lambda_dc_offset: float = 0.0
    lambda_cqt: float = 0.0
    lambda_rail_occupancy: float = 0.0

    mrstft: MRSTFTConfig = field(default_factory=MRSTFTConfig)
    feature_matching: FeatureMatchingLossConfig = field(default_factory=FeatureMatchingLossConfig)
    temporal_gradient: TemporalGradientLossConfig = field(default_factory=TemporalGradientLossConfig)
    rms_envelope: RMSEnvelopeLossConfig = field(default_factory=RMSEnvelopeLossConfig)
    band_power: BandPowerLossConfig = field(default_factory=BandPowerLossConfig)
    spectral_flatness: SpectralFlatnessLossConfig = field(default_factory=SpectralFlatnessLossConfig)
    dc_offset: DCOffsetLossConfig = field(default_factory=DCOffsetLossConfig)
    cqt: CQTConsistencyLossConfig = field(default_factory=CQTConsistencyLossConfig)
    rail_occupancy: RailOccupancyLossConfig = field(default_factory=RailOccupancyLossConfig)


@dataclass
class VocoderLossCalibrationConfig:
    enabled: bool = False

    start_step: int = 5000
    end_step: int = 10000

    output_filename: str = "loss_calibration.json"


@dataclass
class VocoderDataConfig:
    chip_cache_root: str = ""
    sample_rate: int = 22050
    hop_length: int = 512
    cqt_bins: int = 96
    bins_per_octave: int = 12
    fmin: float = 32.70319566

    segment_frames: int = 32
    windows_per_track: int = 8
    batch_size: int = 2
    num_workers: int = 1
    cache_waveforms: int = 8

    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    random_window: bool = True
    drop_last: bool = True


@dataclass
class SnakeBetaConfig:
    alpha_logscale: bool = True
    alpha_initial: float = 1.0
    beta_initial: float = 1.0


@dataclass
class AliasFreeConfig:
    enabled: bool = False
    implementation: str = "pytorch"
    upsample_ratio: int = 2
    downsample_ratio: int = 2
    upsample_kernel_size: int = 12
    downsample_kernel_size: int = 12


@dataclass
class ECAConfig:
    enabled: bool = False

    # 0 == Adaptive
    kernel_size: int = 0

    gamma: float = 2.0
    beta: float = 1.0
    minimum_kernel_size: int = 3

    residual: bool = False


@dataclass
class ContextAttentionConfig:
    enabled: bool = False

    placement: int = 2

    number_of_heads: int = 8
    dropout: float = 0.0

    rope_base: float = 10_000.0

    layer_scale_initial: float = 1e-4

    qkv_bias:bool = True
    output_bias: bool = True


@dataclass
class ContextTrunkConfig:
    enabled: bool = False

    block_type: ContextBlockType = "convnext"
    number_of_blocks: int = 4

    kernel_size: int = 7
    dilation: int = 1

    multi_kernel_sizes: tuple[int, ...] = (3, 7, 15, 31)
    multi_kernel_dilations: tuple[int, ...] = (1, 1, 1, 1)

    expansion_ratio: int = 2
    layer_scale_initial: float = 1e-6

    attention: ContextAttentionConfig = field(default_factory=ContextAttentionConfig)

    use_grn: bool = False
    grn_eps: float = 1e-6


@dataclass
class VocoderGeneratorModelConfig:
    activation: ActivationType = "snake_beta"
    upsample_initial_channel: int = 128

    upsample_rates: tuple[int, ...] = (8, 8, 4, 2)
    upsample_kernel_sizes: tuple[int, ...] = (16, 16, 8, 4)

    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )

    final_tanh: bool = True
    use_weight_norm: bool = True

    snake_beta: SnakeBetaConfig = field(default_factory=SnakeBetaConfig)
    alias_free: AliasFreeConfig = field(default_factory=AliasFreeConfig)

    eca: ECAConfig = field(default_factory=ECAConfig)
    context: ContextTrunkConfig = field(default_factory=ContextTrunkConfig)


@dataclass
class PeriodDiscriminatorConfig:
    period: int
    channels: tuple[int, ...] = (32, 128, 512, 1024, 1024)
    kernel_size: int = 5
    stride: int = 3
    norm: NormType = "weight"
    negative_slope: float = 0.2


@dataclass
class MultiPeriodDiscriminatorConfig:
    periods: tuple[int, ...] = (2, 3, 5, 7, 11)
    channels: tuple[int, ...] = (32, 128, 512, 1024, 1024)
    kernel_size: int = 5
    stride: int = 3
    norm: NormType = "weight"
    negative_slope: float = 0.2


@dataclass
class ScaleDiscriminatorConfig:
    channels: tuple[int, ...] = (128, 128, 256, 512, 1024, 1024)
    kernel_sizes: tuple[int, ...] = (15, 41, 41, 41, 41, 5)
    strides: tuple[int, ...] = (1, 2, 2, 4, 4, 1)
    groups: tuple[int, ...] = (1, 4, 16, 16, 16, 16)
    norm: NormType = "weight"
    negative_slope: float = 0.2


@dataclass
class MultiScaleDiscriminatorConfig:
    discriminator: ScaleDiscriminatorConfig = field(
        default_factory=ScaleDiscriminatorConfig
    )

    num_scales: int = 3

    first_discriminator_norm: NormType = "spectral"
    other_discriminator_norm: NormType = "weight"

    pool_kernel_size: int = 4
    pool_stride: int = 2
    pool_padding: int = 2


@dataclass
class ResolutionDiscriminatorConfig:
    channels: tuple[int, ...] = (
        32,
        64,
        128,
        256,
        512,
    )
    kernel_sizes: tuple[tuple[int, int], ...] = (
        (3, 9),
        (3, 9),
        (3, 9),
        (3, 3),
        (3, 3),
    )
    strides: tuple[tuple[int, int], ...] = (
        (1, 2),
        (2, 2),
        (2, 2),
        (2, 1),
        (1, 1),
    )
    norm: NormType = "weight"
    negative_slope: float = 0.2
    eps: float = 1e-8
    magnitude_compression: str = "log1p"


@dataclass
class MultiResolutionDiscriminatorConfig:
    resolutions: tuple[tuple[int, int, int], ...] = (
        (256, 64, 256),
        (1024, 256, 1024),
        (2048, 512, 2048),
    )

    discriminator: ResolutionDiscriminatorConfig = field(default_factory=ResolutionDiscriminatorConfig)


@dataclass
class VocoderDiscriminatorConfig:
    mpd: MultiPeriodDiscriminatorConfig = field(default_factory=MultiPeriodDiscriminatorConfig)
    msd: MultiScaleDiscriminatorConfig = field(default_factory=MultiScaleDiscriminatorConfig)
    mrd: MultiResolutionDiscriminatorConfig = field(default_factory=MultiResolutionDiscriminatorConfig)

    use_mpd: bool = True
    use_msd: bool = True
    use_mrd: bool = True


@dataclass
class VocoderExperimentConfig:
    schema_version: int = 1

    run: RunConfig = field(default_factory=RunConfig)
    data: VocoderDataConfig = field(default_factory=VocoderDataConfig)
    generator: VocoderGeneratorModelConfig = field(
        default_factory=VocoderGeneratorModelConfig
    )
    discriminator: VocoderDiscriminatorConfig = field(
        default_factory=VocoderDiscriminatorConfig
    )
    loss: VocoderLossConfig = field(
        default_factory=VocoderLossConfig
    )
    optimizer_generator: OptimizerConfig = field(
        default_factory=OptimizerConfig
    )
    optimizer_discriminator: OptimizerConfig = field(
        default_factory=OptimizerConfig
    )
    scheduler_generator: SchedulerConfig = field(
        default_factory=SchedulerConfig
    )
    scheduler_discriminator: SchedulerConfig = field(
        default_factory=SchedulerConfig
    )
    augmentation: AugmentationConfig = field(
        default_factory=AugmentationConfig
    )
    training: TrainingConfig = field(
        default_factory=TrainingConfig
    )
    control: ControlConfig = field(
        default_factory=ControlConfig
    )
    amp: AMPConfig = field(default_factory=AMPConfig)
    logging: LogConfig = field(default_factory=LogConfig)

