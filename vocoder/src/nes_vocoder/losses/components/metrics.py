from __future__ import annotations

import torch

from vocoder.src.nes_vocoder.models.src.losses import FamilyDiscriminatorLossOutput, VocoderGeneratorLossOutput


def generator_loss_metrics(
    output: VocoderGeneratorLossOutput,
) -> dict[str, float]:
    metrics = {
        "loss_g_total": float(
            output.total.detach().cpu()
        ),

        "loss_g_adversarial": float(
            output.adversarial.raw.detach().cpu()
        ),
        "loss_g_feature_matching": float(
            output.feature_matching.raw.detach().cpu()
        ),
        "loss_g_mrstft": float(
            output.mrstft.raw.detach().cpu()
        ),
        "loss_g_waveform": float(
            output.waveform.raw.detach().cpu()
        ),
        "loss_g_temporal_gradient": float(
            output.temporal_gradient.raw.detach().cpu()
        ),
        "loss_g_rms_envelope": float(
            output.rms_envelope.raw.detach().cpu()
        ),
        "loss_g_band_power": float(
            output.band_power.raw.detach().cpu()
        ),
        "loss_g_spectral_flatness": float(
            output.spectral_flatness.raw.detach().cpu()
        ),
        "loss_g_dc_offset": float(
            output.dc_offset.raw.detach().cpu()
        ),
        "loss_g_cqt_consistency": float(
            output.cqt_consistency.raw.detach().cpu()
        ),
        "loss_g_rail_occupancy": float(
            output.rail_occupancy.raw.detach().cpu()
        ),

        "loss_g_weighted/adversarial": float(
            output.adversarial.weighted.detach().cpu()
        ),
        "loss_g_weighted/feature_matching": float(
            output.feature_matching.weighted.detach().cpu()
        ),
        "loss_g_weighted/mrstft": float(
            output.mrstft.weighted.detach().cpu()
        ),
        "loss_g_weighted/waveform": float(
            output.waveform.weighted.detach().cpu()
        ),
        "loss_g_weighted/temporal_gradient": float(
            output.temporal_gradient.weighted.detach().cpu()
        ),
        "loss_g_weighted/rms_envelope": float(
            output.rms_envelope.weighted.detach().cpu()
        ),
        "loss_g_weighted/band_power": float(
            output.band_power.weighted.detach().cpu()
        ),
        "loss_g_weighted/spectral_flatness": float(
            output.spectral_flatness.weighted.detach().cpu()
        ),
        "loss_g_weighted/dc_offset": float(
            output.dc_offset.weighted.detach().cpu()
        ),
        "loss_g_weighted/cqt_consistency": float(
            output.cqt_consistency.weighted.detach().cpu()
        ),
        "loss_g_weighted/rail_occupancy": float(
            output.rail_occupancy.weighted.detach().cpu()
        ),

        "loss_g_group/learned": float(
            output.learned_total.detach().cpu()
        ),
        "loss_g_group/time_micro": float(
            output.time_micro_total.detach().cpu()
        ),
        "loss_g_group/time_frequency": float(
            output.time_frequency_total.detach().cpu()
        ),
        "loss_g_group/time_macro": float(
            output.time_macro_total.detach().cpu()
        ),
        "loss_g_group/spectral_macro": float(
            output.spectral_macro_total.detach().cpu()
        ),

        "loss_g_mrstft_spectral_convergence": float(
            output.mrstft_output
            .spectral_convergence
            .detach()
            .cpu()
        ),
        "loss_g_mrstft_log_magnitude": float(
            output.mrstft_output
            .log_magnitude
            .detach()
            .cpu()
        ),
    }

    metrics.update(
        generator_loss_contribution_metrics(
            output=output,
        )
    )

    for family, value in (
            output.adversarial_families.items()
    ):
        metrics[
            f"loss_g_adversarial_{family}"
        ] = float(
            value.detach().cpu()
        )

    for family, value in (
            output.feature_matching_families.items()
    ):
        metrics[
            f"loss_g_feature_matching_{family}"
        ] = float(
            value.detach().cpu()
        )

    for resolution in (
        output.mrstft_output.resolutions
    ):
        prefix = (
            "loss_g_mrstft_resolution/"
            f"fft_{resolution.fft_size}"
            f"_hop_{resolution.hop_size}"
            f"_win_{resolution.win_length}"
        )

        metrics[
            f"{prefix}/spectral_convergence"
        ] = float(
            resolution
            .spectral_convergence
            .detach()
            .cpu()
        )

        metrics[
            f"{prefix}/log_magnitude"
        ] = float(
            resolution
            .log_magnitude
            .detach()
            .cpu()
        )

        metrics[
            f"{prefix}/total"
        ] = float(
            resolution.total
            .detach()
            .cpu()
        )

        metrics[f"{prefix}/reference_norm_mean"] = float(
            resolution.reference_norm_mean.detach().cpu()
        )

        metrics[f"{prefix}/activity_mean"] = float(
            resolution.activity_mean.detach().cpu()
        )

        metrics[f"{prefix}/floor_activation_fraction"] = float(
            resolution.floor_activation_fraction.detach().cpu()
        )

    if (
            output.temporal_gradient_output
            is not None
    ):
        temporal_output = (
            output.temporal_gradient_output
        )

        metrics[
            "loss_g_slew_envelope_mean"
        ] = float(
            temporal_output.mean
            .detach()
            .cpu()
        )

        metrics[
            "loss_g_slew_envelope_peak"
        ] = float(
            temporal_output.peak
            .detach()
            .cpu()
        )

        for scale in temporal_output.scales:
            prefix = (
                "loss_g_slew_envelope_scale/"
                f"lag_{scale.lag}"
                f"_window_{scale.window_size}"
                f"_hop_{scale.hop_size}"
            )

            metrics[
                f"{prefix}/mean"
            ] = float(
                scale.mean.detach().cpu()
            )

            metrics[
                f"{prefix}/peak"
            ] = float(
                scale.peak.detach().cpu()
            )

            metrics[
                f"{prefix}/total"
            ] = float(
                scale.total.detach().cpu()
            )

    if output.rms_envelope_output is not None:
        for scale in output.rms_envelope_output.scales:
            prefix = (
                "loss_g_rms_envelope_scale/"
                f"window_{scale.window_size}"
                f"_hop_{scale.hop_size}"
            )

            metrics[prefix] = float(
                scale.loss.detach().cpu()
            )

    if output.band_power_output is not None:
        metrics["loss_g_band_power_absolute"] = float(
            output.band_power_output.absolute.detach().cpu()
        )
        metrics["loss_g_band_power_shape"] = float(
            output.band_power_output.shape.detach().cpu()
        )

        for resolution in output.band_power_output.resolutions:
            prefix = (
                "loss_g_band_power_resolution/"
                f"fft_{resolution.fft_size}"
                f"_hop_{resolution.hop_size}"
                f"_win_{resolution.win_length}"
            )

            metrics[f"{prefix}/absolute"] = float(
                resolution.absolute.detach().cpu()
            )

            metrics[f"{prefix}/shape"] = float(
                resolution.shape.detach().cpu()
            )

            metrics[f"{prefix}/total"] = float(
                resolution.total.detach().cpu()
            )

            metrics[f"{prefix}/activity_mean"] = float(
                resolution.activity_mean.detach().cpu()
            )

            metrics[f"{prefix}/floor_activation_fraction"] = float(
                resolution.floor_activation_fraction.detach().cpu()
            )

    if output.spectral_flatness_output is not None:
        for band_index, band in enumerate(
                output
                        .spectral_flatness_output
                        .bands
        ):
            prefix = (
                "loss_g_spectral_flatness_band/"
                f"band_{band_index}"
                f"_{band.lower_frequency:g}"
                f"_{band.upper_frequency:g}"
            )

            metrics[
                f"{prefix}/loss"
            ] = float(
                band.loss.detach().cpu()
            )

            metrics[
                f"{prefix}/target_log_flatness"
            ] = float(
                band.target_log_flatness
                .detach()
                .cpu()
            )

            metrics[
                f"{prefix}/prediction_log_flatness"
            ] = float(
                band.prediction_log_flatness
                .detach()
                .cpu()
            )

            metrics[
                f"{prefix}/activity"
            ] = float(
                band.activity_fraction
                .detach()
                .cpu()
            )

            metrics[f"{prefix}/floor_activation_fraction"] = float(
                band.floor_activation_fraction.detach().cpu()
            )

    if output.dc_offset_output is not None:
        dc_output = output.dc_offset_output

        metrics[
            "loss_g_dc_offset_local"
        ] = float(
            dc_output.local
            .detach()
            .cpu()
        )

        metrics[
            "loss_g_dc_offset_global"
        ] = float(
            dc_output.global_mean
            .detach()
            .cpu()
        )

        for scale in dc_output.scales:
            prefix = (
                "loss_g_dc_offset_scale/"
                f"window_{scale.window_size}"
                f"_hop_{scale.hop_size}"
            )

            metrics[prefix] = float(
                scale.loss.detach().cpu()
            )

    if output.cqt_consistency_output is not None:
        cqt_output = output.cqt_consistency_output

        metrics["loss_g_cqt_log_magnitude"] = float(
            cqt_output.log_magnitude.detach().cpu()
        )

        metrics["cqt/prediction_log_magnitude_mean"] = float(
            cqt_output.prediction_log_magnitude_mean.detach().cpu()
        )

        metrics["cqt/target_log_magnitude_mean"] = float(
            cqt_output.target_log_magnitude_mean.detach().cpu()
        )

    rail_output = output.rail_occupancy_output

    if rail_output is not None:
        metrics[
            "rail_occupancy/target_mean"
        ] = float(
            rail_output.target_occupancy
            .detach()
        .cpu()
        )

        metrics[
            "rail_occupancy/prediction_mean"
        ] = float(
            rail_output.prediction_occupancy
            .detach()
            .cpu()
        )

        for scale in rail_output.scales:
            prefix = (
                "loss_g_rail_occupancy_scale_"
                f"window_{scale.window_size}"
                f"_hop_{scale.hop_size}"
            )

            metrics[
                f"{prefix}/loss"
            ] = float(
                scale.loss.detach().cpu()
            )

            metrics[
                f"{prefix}/target"
            ] = float(
                scale.target_occupancy
                .detach()
                .cpu()
            )

            metrics[
                f"{prefix}/prediction"
            ] = float(
                scale.prediction_occupancy
                .detach()
                .cpu()
            )

            metrics[
                f"{prefix}/target_peak"
            ] = float(
                scale.target_peak_mean
                .detach()
                .cpu()
            )

            metrics[
                f"{prefix}/activity"
            ] = float(
                scale.activity_fraction
                .detach()
                .cpu()
            )

    return metrics


def discriminator_loss_metrics(
    output: FamilyDiscriminatorLossOutput,
) -> dict[str, float]:
    metrics = {
        "loss_d_total": float(
            output.total.detach().cpu()
        ),
        "loss_d_real": float(
            output.real.detach().cpu()
        ),
        "loss_d_fake": float(
            output.fake.detach().cpu()
        ),
    }

    for family, value in (
        output.family_totals.items()
    ):
        metrics[
            f"loss_d_{family}_total"
        ] = float(
            value.detach().cpu()
        )

    for family, value in (
        output.family_real.items()
    ):
        metrics[
            f"loss_d_{family}_real"
        ] = float(
            value.detach().cpu()
        )

    for family, value in (
        output.family_fake.items()
    ):
        metrics[
            f"loss_d_{family}_fake"
        ] = float(
            value.detach().cpu()
        )

    return metrics


def generator_loss_contribution_metrics(
    output: VocoderGeneratorLossOutput,
    *,
    eps: float = 1e-8,
) -> dict[str, float]:
    weighted_components = {
        "adversarial": (
            output.adversarial.weighted
        ),
        "feature_matching": (
            output.feature_matching.weighted
        ),
        "mrstft": output.mrstft.weighted,
        "waveform": output.waveform.weighted,
        "temporal_gradient": (
            output.temporal_gradient.weighted
        ),
        "rms_envelope": (
            output.rms_envelope.weighted
        ),
        "band_power": (
            output.band_power.weighted
        ),
        "spectral_flatness": (
            output.spectral_flatness.weighted
        ),
        "dc_offset": (
            output.dc_offset.weighted
        ),
        "cqt_consistency": (
            output.cqt_consistency.weighted
        ),
        "rail_occupancy": (
            output.rail_occupancy.weighted
        )
    }

    denominator = torch.stack(
        [
            value.detach().abs()
            for value in weighted_components.values()
        ]
    ).sum().clamp_min(eps)

    metrics: dict[str, float] = {}

    for name, value in weighted_components.items():
        contribution = (
            value.detach().abs()
            / denominator
        )

        metrics[
            f"loss_g_contribution/{name}"
        ] = float(
            contribution.cpu()
        )

    return metrics
