from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from vocoder.src.nes_vocoder.config.schema import VocoderLossConfig
from src.losses.vocoder.adversarial_loss import family_balanced_generator_adversarial_loss
from src.losses.vocoder.band_power_loss import MultiResolutionBandPowerLoss, MultiResolutionBandPowerLossOutput
from src.losses.vocoder.cqt_consistency_loss import CQTLogMagnitudeLoss, CQTConsistencyLossOutput
from src.losses.vocoder.dc_offset_loss import MultiScaleDCOffsetLoss, MultiScaleDCOffsetLossOutput
from src.losses.vocoder.feature_matching_loss import DISCRIMINATOR_FAMILIES, mean_enabled_families, \
    family_balanced_feature_matching_loss
from src.losses.vocoder.mrstft_loss import MultiResolutionSTFTLoss, MultiResolutionSTFTLossOutput
from src.losses.vocoder.rail_occupancy_loss import MultiScaleRailOccupancyLoss, \
    MultiScaleRailOccupancyLossOutput
from src.losses.vocoder.rms_envelope_loss import MultiScaleRMSEnvelopeLoss, MultiScaleRMSEnvelopeLossOutput
from src.losses.vocoder.spectral_flatness_loss import SubBandSpectralFlatnessLoss, SubBandSpectralFlatnessLossOutput
from src.losses.vocoder.temporal_gradient_loss import MultiScaleTemporalGradientLoss, \
    MultiScaleSlewEnvelopeLoss, MultiScaleSlewEnvelopeLossOutput
from src.losses.vocoder.waveform_loss import waveform_loss


@dataclass
class VocoderDiscriminatorLossOutput:
    total: torch.Tensor
    real: torch.Tensor
    fake: torch.Tensor


class VocoderLossBundle(nn.Module):
    """
    Container for vocoder losses.

    Kept as nn.Module so the STFT loss modules are properly moved with .to(device)
    if needed, although windows are created device-locally at runtime.
    """

    def __init__(
        self,
        config: VocoderLossConfig = VocoderLossConfig(),
        *,
        sample_rate: int,
    ) -> None:
        super().__init__()

        self.config = config
        self.mrstft_loss = MultiResolutionSTFTLoss(config.mrstft)
        self.temporal_gradient_loss: (
                nn.Module | None
        )

        if not config.temporal_gradient.enabled:
            self.temporal_gradient_loss = None

        elif config.temporal_gradient.mode == "pointwise_gradient":
            self.temporal_gradient_loss = (
                MultiScaleTemporalGradientLoss(
                    lags=config.temporal_gradient.lags,
                    normalization=config.temporal_gradient.normalization,
                    scale_floor=config.temporal_gradient.scale_floor,
                    eps=config.temporal_gradient.eps,
                    edge_weighting=config.temporal_gradient.edge_weighting,
                    edge_weight=config.temporal_gradient.edge_weight,
                    max_edge_weight=config.temporal_gradient.max_edge_weight,
                )
            )
        elif config.temporal_gradient.mode == "slew_envelope":
            self.temporal_gradient_loss = (
                MultiScaleSlewEnvelopeLoss(
                    lags=config.temporal_gradient.lags,
                    window_sizes=config.temporal_gradient.window_sizes,
                    hop_ratio=config.temporal_gradient.hop_ratio,
                    mean_weight=config.temporal_gradient.mean_weight,
                    peak_weight=config.temporal_gradient.peak_weight,
                )
            )
        else:
            raise ValueError (
            "Unsupported temporal-gradient mode: "
            f"{config.temporal_gradient.mode!r}."

        )
        self.rms_envelope_loss = MultiScaleRMSEnvelopeLoss(
            config.rms_envelope
        ) if config.rms_envelope.enabled else None
        self.band_power_loss = MultiResolutionBandPowerLoss(
            sample_rate=sample_rate,
            config=config.band_power,
        ) if config.band_power.enabled else None
        self.spectral_flatness_loss = SubBandSpectralFlatnessLoss(
            sample_rate=sample_rate,
            config=config.spectral_flatness,
        ) if config.spectral_flatness.enabled else None
        self.dc_offset_loss = (
            MultiScaleDCOffsetLoss(
                config.dc_offset
            )
            if config.dc_offset.enabled
            else None
        )
        self.cqt_consistency_loss = CQTLogMagnitudeLoss(
            loss_config=config.cqt
        ) if config.cqt.enabled else None
        self.rail_occupancy_loss = MultiScaleRailOccupancyLoss(
            config=config.rail_occupancy,
        ) if config.rail_occupancy.enabled else None


def lsgan_discriminator_loss(
    real_outputs: list[torch.Tensor],
    fake_outputs: list[torch.Tensor],
) -> VocoderDiscriminatorLossOutput:
    if not real_outputs:
        raise ValueError("No discriminator outputs supplied.")

    real_losses = [
        torch.mean((1.0 - output) ** 2)
        for output in real_outputs
    ]

    fake_losses = [
        torch.mean(output ** 2)
        for output in fake_outputs
    ]

    loss_real = torch.stack(real_losses).mean()
    loss_fake = torch.stack(fake_losses).mean()

    return VocoderDiscriminatorLossOutput(
        loss_real + loss_fake,
        loss_real,
        loss_fake,
    )


@dataclass(frozen=True)
class FamilyDiscriminatorLossOutput:
    total: torch.Tensor
    real: torch.Tensor
    fake: torch.Tensor

    family_totals: dict[str, torch.Tensor]
    family_real: dict[str, torch.Tensor]
    family_fake: dict[str, torch.Tensor]


def family_balanced_discriminator_loss(
    discriminator_outputs: dict[str, list],
) -> FamilyDiscriminatorLossOutput:
    family_totals: dict[
        str,
        torch.Tensor,
    ] = {}

    family_real: dict[
        str,
        torch.Tensor,
    ] = {}

    family_fake: dict[
        str,
        torch.Tensor,
    ] = {}

    for family in DISCRIMINATOR_FAMILIES:
        real_outputs = discriminator_outputs[
            f"{family}_real_outputs"
        ]

        fake_outputs = discriminator_outputs[
            f"{family}_fake_outputs"
        ]

        if not real_outputs:
            continue

        family_output = (
            lsgan_discriminator_loss(
                real_outputs=real_outputs,
                fake_outputs=fake_outputs,
            )
        )

        family_totals[family] = (
            family_output.total
        )

        family_real[family] = (
            family_output.real
        )

        family_fake[family] = (
            family_output.fake
        )

    total = mean_enabled_families(
        family_totals,
        name="discriminator total loss",
    )

    real = mean_enabled_families(
        family_real,
        name="discriminator real loss",
    )

    fake = mean_enabled_families(
        family_fake,
        name="discriminator fake loss",
    )

    return FamilyDiscriminatorLossOutput(
        total=total,
        real=real,
        fake=fake,
        family_totals=family_totals,
        family_real=family_real,
        family_fake=family_fake,
    )


@dataclass
class VocoderGeneratorLossOutput:
    total: torch.Tensor

    adversarial: WeightedLossComponent
    feature_matching: WeightedLossComponent
    mrstft: WeightedLossComponent
    waveform: WeightedLossComponent
    temporal_gradient: WeightedLossComponent
    rms_envelope: WeightedLossComponent
    band_power: WeightedLossComponent
    spectral_flatness: WeightedLossComponent
    dc_offset: WeightedLossComponent
    cqt_consistency: WeightedLossComponent
    rail_occupancy: WeightedLossComponent

    mrstft_output: MultiResolutionSTFTLossOutput
    temporal_gradient_output: MultiScaleSlewEnvelopeLossOutput | None
    rms_envelope_output: MultiScaleRMSEnvelopeLossOutput | None
    band_power_output: MultiResolutionBandPowerLossOutput | None
    spectral_flatness_output: SubBandSpectralFlatnessLossOutput | None
    dc_offset_output: MultiScaleDCOffsetLossOutput | None
    cqt_consistency_output: CQTConsistencyLossOutput | None
    rail_occupancy_output: MultiScaleRailOccupancyLossOutput | None

    adversarial_families: dict[str, torch.Tensor]
    feature_matching_families: dict[str, torch.Tensor]

    @property
    def learned_total(self) -> torch.Tensor:
        return self.adversarial.weighted + self.feature_matching.weighted

    @property
    def time_micro_total(self) -> torch.Tensor:
        return self.waveform.weighted + self.temporal_gradient.weighted + self.rail_occupancy.weighted

    @property
    def time_frequency_total(self) -> torch.Tensor:
        return self.mrstft.weighted + self.cqt_consistency.weighted

    @property
    def time_macro_total(self) -> torch.Tensor:
        return self.rms_envelope.weighted + self.dc_offset.weighted

    @property
    def spectral_macro_total(self) -> torch.Tensor:
        return self.band_power.weighted + self.spectral_flatness.weighted


def compute_vocoder_generator_loss(
    discriminator_outputs: dict[str, list],
    fake_audio: torch.Tensor,
    real_audio: torch.Tensor,
    loss_bundle: VocoderLossBundle,
) -> VocoderGeneratorLossOutput:
    """
    Computes full generator loss
    """
    cfg = loss_bundle.config

    #------------------Raw losses-------------------

    adversarial_output = (
        family_balanced_generator_adversarial_loss(
            discriminator_outputs
        )
    )

    feature_config = cfg.feature_matching

    feature_matching_output = (
        family_balanced_feature_matching_loss(
            discriminator_outputs,
            normalization=(
                feature_config.normalization
            ),
            scale_floor=(
                feature_config.scale_floor
            ),
            eps=feature_config.eps,
        )
    )

    loss_adversarial_raw = (
        adversarial_output.total
    )

    loss_feature_matching_raw = (
        feature_matching_output.total
    )

    mrstft_output = loss_bundle.mrstft_loss(
        fake_audio,
        real_audio,
    )

    loss_mrstft_raw = mrstft_output.total

    loss_waveform_raw = waveform_loss(
        fake_audio,
        real_audio,
    )

    temporal_gradient_output = None

    if loss_bundle.temporal_gradient_loss is None:
        loss_temporal_gradient_raw = fake_audio.new_zeros(())


    elif isinstance(loss_bundle.temporal_gradient_loss, MultiScaleSlewEnvelopeLoss):
        temporal_gradient_output = loss_bundle.temporal_gradient_loss(
            fake_audio,
            real_audio,
        )

        loss_temporal_gradient_raw = temporal_gradient_output.total

    else:
        loss_temporal_gradient_raw = loss_bundle.temporal_gradient_loss(
            fake_audio,
            real_audio,
        )

    if loss_bundle.rms_envelope_loss is None:
        rms_envelope_output = None

        loss_rms_envelope_raw = fake_audio.new_zeros(())
    else:
        rms_envelope_output = loss_bundle.rms_envelope_loss(fake_audio, real_audio)

        loss_rms_envelope_raw = rms_envelope_output.total

    if loss_bundle.band_power_loss is None:
        band_power_output = None

        loss_band_power_raw = fake_audio.new_zeros(())
    else:
        band_power_output = loss_bundle.band_power_loss(fake_audio, real_audio)

        loss_band_power_raw = band_power_output.total

    if loss_bundle.spectral_flatness_loss is None:
        spectral_flatness_output = None

        loss_spectral_flatness_raw = fake_audio.new_zeros(())
    else:
        spectral_flatness_output = loss_bundle.spectral_flatness_loss(
            fake_audio,
            real_audio,
        )

        loss_spectral_flatness_raw = spectral_flatness_output.total

    if loss_bundle.dc_offset_loss is None:
        dc_offset_output = None

        loss_dc_offset_raw = (
            fake_audio.new_zeros(())
        )
    else:
        dc_offset_output = (
            loss_bundle.dc_offset_loss(
                fake_audio,
                real_audio,
            )
        )

        loss_dc_offset_raw = (
            dc_offset_output.total
        )

    if loss_bundle.cqt_consistency_loss is None:
        cqt_output = None

        loss_cqt_raw = fake_audio.new_zeros(())
    else:
        cqt_output = loss_bundle.cqt_consistency_loss(
            fake_audio,
            real_audio,
        )

        loss_cqt_raw = cqt_output.total

    if loss_bundle.rail_occupancy_loss is None:
        rail_occupancy_output = None

        loss_rail_occupancy_raw = fake_audio.new_zeros(())
    else:
        rail_occupancy_output = loss_bundle.rail_occupancy_loss(
            fake_audio,
            real_audio,
        )

        loss_rail_occupancy_raw = rail_occupancy_output.total

    #------------------Weighted losses-------------------

    adversarial = make_weighted_loss_component(
        raw=loss_adversarial_raw,
        weight=cfg.lambda_adversarial,
    )

    feature_matching = make_weighted_loss_component(
        raw=loss_feature_matching_raw,
        weight=cfg.lambda_feature_matching,
    )

    mrstft = make_weighted_loss_component(
        raw=loss_mrstft_raw,
        weight=cfg.lambda_mrstft,
    )

    waveform = make_weighted_loss_component(
        raw=loss_waveform_raw,
        weight=cfg.lambda_waveform,
    )

    temporal_gradient = make_weighted_loss_component(
        raw=loss_temporal_gradient_raw,
        weight=cfg.lambda_temporal_gradient,
    )

    rms_envelope = make_weighted_loss_component(
        raw=loss_rms_envelope_raw,
        weight=cfg.lambda_rms_envelope,
    )

    band_power = make_weighted_loss_component(
        raw=loss_band_power_raw,
        weight=cfg.lambda_band_power,
    )

    spectral_flatness = make_weighted_loss_component(
        raw=loss_spectral_flatness_raw,
        weight=cfg.lambda_spectral_flatness,
    )

    dc_offset = make_weighted_loss_component(
        raw=loss_dc_offset_raw,
        weight=cfg.lambda_dc_offset,
    )

    cqt_consistency = make_weighted_loss_component(
        raw=loss_cqt_raw,
        weight=cfg.lambda_cqt,
    )

    rail_occupancy = make_weighted_loss_component(
        raw=loss_rail_occupancy_raw,
        weight=cfg.lambda_rail_occupancy,
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
        feature_matching=feature_matching,
        mrstft=mrstft,
        waveform=waveform,
        temporal_gradient=temporal_gradient,
        rms_envelope=rms_envelope,
        band_power=band_power,
        spectral_flatness=spectral_flatness,
        dc_offset=dc_offset,
        cqt_consistency=cqt_consistency,
        rail_occupancy=rail_occupancy,
        mrstft_output=mrstft_output,
        temporal_gradient_output=temporal_gradient_output,
        rms_envelope_output=rms_envelope_output,
        band_power_output=band_power_output,
        adversarial_families=adversarial_output.families,
        feature_matching_families=feature_matching_output.families,
        spectral_flatness_output=spectral_flatness_output,
        dc_offset_output=dc_offset_output,
        cqt_consistency_output=cqt_output,
        rail_occupancy_output=rail_occupancy_output,
    )


def discriminator_family_losses(
    discriminator_outputs: dict[str, list],
) -> dict[str, VocoderDiscriminatorLossOutput]:
    family_losses: dict[
        str,
        VocoderDiscriminatorLossOutput,
    ] = {}

    for family in ("mpd", "msd", "mrd"):
        real_outputs = discriminator_outputs[
            f"{family}_real_outputs"
        ]
        fake_outputs = discriminator_outputs[
            f"{family}_fake_outputs"
        ]

        if not real_outputs:
            continue

        family_losses[family] = (
            lsgan_discriminator_loss(
                real_outputs=real_outputs,
                fake_outputs=fake_outputs,
            )
        )

    return family_losses


def compute_vocoder_discriminator_loss(
    discriminator_outputs: dict[str, list],
) -> VocoderDiscriminatorLossOutput:
    """
    Computes discriminator LSGAN loss from combined MPD + MSD outputs.
    """
    return lsgan_discriminator_loss(
        real_outputs=discriminator_outputs["real_outputs"],
        fake_outputs=discriminator_outputs["fake_outputs"],
    )


def detach_vocoder_loss_dict(losses: dict[str, torch.Tensor | float]) -> dict[str, float]:
    out: dict[str, float] = {}

    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            out[key] = float(value.detach().cpu().item())
        else:
            out[key] = float(value)

    return out


@dataclass(frozen=True)
class WeightedLossComponent:
    """
    One generator loss before and after application of its configured scalar weight.
    """

    raw: torch.Tensor
    weight: float
    weighted: torch.Tensor


def make_weighted_loss_component(
        raw: torch.Tensor,
        weight: float,
) -> WeightedLossComponent:
    return WeightedLossComponent(
        raw=raw,
        weight=weight,
        weighted=raw * float(weight)
    )
