import math

import pytest
import torch
import torch.nn.functional as F

from vocoder.src.nes_vocoder.config.schema import MRSTFTConfig, RMSEnvelopeLossConfig, BandPowerLossConfig, \
    SpectralFlatnessLossConfig, \
    DCOffsetLossConfig, VocoderLossConfig
from vocoder.src.nes_vocoder.losses.components.adversarial_loss import family_balanced_generator_adversarial_loss
from vocoder.src.nes_vocoder.losses.components.band_power_loss import build_log_frequency_filterbank, \
    MultiResolutionBandPowerLoss
from vocoder.src.nes_vocoder.losses.components.dc_offset_loss import MultiScaleDCOffsetLoss
from vocoder.src.nes_vocoder.losses.components.feature_matching_loss import feature_matching_loss, \
    mean_enabled_families, family_balanced_feature_matching_loss
from vocoder.src.nes_vocoder.losses.components.metrics import generator_loss_contribution_metrics, \
    generator_loss_metrics, discriminator_loss_metrics
from vocoder.src.nes_vocoder.losses.components.mrstft_loss import MultiResolutionSTFTLoss, SingleResolutionSTFTLoss
from vocoder.src.nes_vocoder.losses.components.rms_envelope_loss import MultiScaleRMSEnvelopeLoss
from vocoder.src.nes_vocoder.losses.components.spectral_flatness_loss import log_spectral_flatness, \
    SubBandSpectralFlatnessLoss
from vocoder.src.nes_vocoder.losses.components.temporal_gradient_loss import MultiScaleTemporalGradientLoss, \
    MultiScaleSlewEnvelopeLoss
from vocoder.src.nes_vocoder.losses.objective import VocoderLossBundle, compute_vocoder_discriminator_loss, \
    family_balanced_discriminator_loss
from vocoder.tests.helpers.vocoder_loss_helpers import make_test_generator_loss_output, make_discriminator_family, \
    make_multi_family_discriminator_outputs, assert_family_loss_output_consistent, \
    assert_family_discriminator_loss_consistent, make_test_spectral_flatness_loss, make_test_cqt_loss, \
    make_test_rail_occupancy_loss


def test_mrstft_loss_finite():
    bundle = VocoderLossBundle(
        VocoderLossConfig(
            mrstft=MRSTFTConfig(
                fft_sizes=(256, 512),
                hop_sizes=(64, 128),
                win_lengths=(256, 512),
            )
        ),
        sample_rate=8000
    )

    real = torch.randn(2, 1, 4096)
    fake = torch.randn(2, 1, 4096)

    loss = bundle.mrstft_loss(fake, real).total
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_vocoder_discriminator_loss_finite():
    outputs = {
        "real_outputs": [torch.ones(2, 8), torch.ones(2, 4)],
        "fake_outputs": [torch.zeros(2, 8), torch.zeros(2, 4)],
    }

    losses = compute_vocoder_discriminator_loss(outputs)

    assert losses.total.ndim == 0
    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.real)
    assert torch.isfinite(losses.fake)


def test_feature_matching_averages_layers_equally() -> None:
    real_maps = [
        [
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 100),
        ]
    ]

    fake_maps = [
        [
            torch.ones(1, 1, 2),
            torch.full((1, 1, 100), 3.0),
        ]
    ]

    loss = feature_matching_loss(real_maps, fake_maps, normalization="layer_mean")
    assert loss.item() == pytest.approx(2.0)


def test_feature_matching_normalizes_real_scale() -> None:
    real_maps = [
        [
            torch.full((1, 1, 8), 2.0)
        ]
    ]

    fake_maps = [
        [
            torch.full((1, 1, 8), 3.0)
        ]
    ]

    loss = feature_matching_loss(real_maps, fake_maps, normalization="real_magnitude", scale_floor=0.01)

    assert loss.item() == pytest.approx(0.5)


def test_feature_matching_normalizes_by_real_rms() -> None:
    real_maps = [
        [
            torch.tensor([[[0.0, 2.0]]])
        ]
    ]
    fake_maps = [
        [
            torch.tensor([[[1.0, 3.0]]])
        ]
    ]

    loss = feature_matching_loss(
        real_maps,
        fake_maps,
        normalization="real_rms",
        scale_floor=0.01,
        eps=1e-8,
    )

    assert loss.item() == pytest.approx(1.0 / math.sqrt(2.0))


def test_feature_matching_real_rms_uses_scale_floor() -> None:
    real_maps = [[torch.zeros(1, 1, 8)]]
    fake_maps = [[torch.ones(1, 1, 8)]]

    loss = feature_matching_loss(
        real_maps,
        fake_maps,
        normalization="real_rms",
        scale_floor=0.25,
        eps=1e-8,
    )

    assert loss.item() == pytest.approx(4.0)


def test_temporal_gradient_zero_for_exact_match() -> None:
    loss_module = MultiScaleTemporalGradientLoss(
        lags=(1, 2, 4)
    )

    waveform = torch.randn(2, 1, 128)

    loss = loss_module(waveform, waveform)

    assert loss.item() == pytest.approx(0.0)


def test_temporal_gradient_is_polarity_sensitive() -> None:
    loss_module = MultiScaleTemporalGradientLoss(
        lags=(1, 2, 4)
    )

    waveform = torch.randn(2, 1, 128)

    loss = loss_module(-waveform, waveform)

    assert loss.item() > 0.0


def test_temporal_gradient_backpropagates() -> None:
    loss_module = MultiScaleTemporalGradientLoss(
        lags=(1, 2, 4)
    )

    prediction = torch.randn(2, 1, 128, requires_grad=True)

    target = torch.randn(2, 1, 128)

    loss = loss_module(prediction, target)

    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_temporal_gradient_prefers_correct_edges() -> None:
    target = torch.tensor(
        [
            [
                [
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                ]
            ]
        ]
    )

    correct = target.clone()

    smoothed = torch.tensor(
        [
            [
                [
                    -1.0,
                    -0.8,
                    -0.2,
                    0.4,
                    1.0,
                    0.5,
                    -0.3,
                    -1.0,
                ]
            ]
        ]
    )

    loss_module = MultiScaleTemporalGradientLoss(
        lags=(1, 2)
    )

    correct_loss = loss_module(correct, target)
    smoothed_loss = loss_module(smoothed, target)

    assert correct_loss < smoothed_loss


def test_mrstft_components_are_zero_for_exact_match() -> None:
    config = MRSTFTConfig(
        fft_sizes=(128, 256),
        hop_sizes=(32, 64),
        win_lengths=(128, 256),
        spectral_convergence_weight=1.0,
        log_magnitude_weight=1.0,
        eps=1e-7,
    )

    loss_function = MultiResolutionSTFTLoss(
        config
    )

    waveform = torch.randn(
        2,
        1,
        2048,
    )

    output = loss_function(
        waveform,
        waveform,
    )

    assert output.total.item() == pytest.approx(
        0.0,
        abs=1e-6,
    )

    assert (
        output.spectral_convergence.item()
        == pytest.approx(
            0.0,
            abs=1e-6,
        )
    )

    assert (
        output.log_magnitude.item()
        == pytest.approx(
            0.0,
            abs=1e-6,
        )
    )

    assert len(output.resolutions) == 2


def test_mrstft_total_matches_components() -> None:
    config = MRSTFTConfig(
        fft_sizes=(128, 256),
        hop_sizes=(32, 64),
        win_lengths=(128, 256),
        spectral_convergence_weight=2.0,
        log_magnitude_weight=0.5,
        eps=1e-7,
    )

    loss_function = MultiResolutionSTFTLoss(
        config
    )

    fake = torch.randn(2, 1, 2048)
    real = torch.randn(2, 1, 2048)

    output = loss_function(fake, real)

    expected = (
        config.spectral_convergence_weight
        * output.spectral_convergence
        + config.log_magnitude_weight
        * output.log_magnitude
    )

    assert torch.allclose(
        output.total,
        expected,
        atol=1e-7,
        rtol=1e-6,
    )


def test_mrstft_reports_every_resolution() -> None:
    config = MRSTFTConfig(
        fft_sizes=(128, 256, 512),
        hop_sizes=(32, 64, 128),
        win_lengths=(128, 256, 512),
    )

    loss_function = MultiResolutionSTFTLoss(
        config
    )

    fake = torch.randn(1, 1, 2048)
    real = torch.randn(1, 1, 2048)

    output = loss_function(fake, real)

    metadata = [
        (
            resolution.fft_size,
            resolution.hop_size,
            resolution.win_length,
        )
        for resolution in output.resolutions
    ]

    assert metadata == [
        (128, 32, 128),
        (256, 64, 256),
        (512, 128, 512),
    ]


def test_mrstft_means_match_resolution_outputs() -> None:
    config = MRSTFTConfig(
        fft_sizes=(128, 256),
        hop_sizes=(32, 64),
        win_lengths=(128, 256),
    )

    loss_function = MultiResolutionSTFTLoss(
        config
    )

    fake = torch.randn(1, 1, 2048)
    real = torch.randn(1, 1, 2048)

    output = loss_function(fake, real)

    expected_convergence = torch.stack(
        [
            resolution.spectral_convergence
            for resolution in output.resolutions
        ]
    ).mean()

    expected_log_magnitude = torch.stack(
        [
            resolution.log_magnitude
            for resolution in output.resolutions
        ]
    ).mean()

    assert torch.allclose(
        output.spectral_convergence,
        expected_convergence,
    )

    assert torch.allclose(
        output.log_magnitude,
        expected_log_magnitude,
    )


def test_mrstft_structured_output_backpropagates() -> None:
    config = MRSTFTConfig(
        fft_sizes=(128, 256),
        hop_sizes=(32, 64),
        win_lengths=(128, 256),
    )

    loss_function = MultiResolutionSTFTLoss(
        config
    )

    fake = torch.randn(
        2,
        1,
        2048,
        requires_grad=True,
    )

    real = torch.randn(
        2,
        1,
        2048,
    )

    output = loss_function(fake, real)

    output.total.backward()

    assert fake.grad is not None
    assert torch.isfinite(fake.grad).all()
    assert fake.grad.abs().sum() > 0


def test_structured_mrstft_matches_legacy_scalar() -> None:
    config = MRSTFTConfig(
        fft_sizes=(128, 256),
        hop_sizes=(32, 64),
        win_lengths=(128, 256),
        spectral_convergence_weight=1.3,
        log_magnitude_weight=0.7,
    )

    loss_function = MultiResolutionSTFTLoss(
        config
    )

    fake = torch.randn(2, 1, 2048)
    real = torch.randn(2, 1, 2048)

    output = loss_function(fake, real)

    legacy_resolution_totals = []

    for single_loss in loss_function.losses:
        convergence, log_magnitude, _, _, _ = (
            single_loss(fake, real)
        )

        legacy_resolution_totals.append(
            config.spectral_convergence_weight
            * convergence
            + config.log_magnitude_weight
            * log_magnitude
        )

    legacy_total = torch.stack(
        legacy_resolution_totals
    ).mean()

    assert torch.allclose(
        output.total,
        legacy_total,
        atol=1e-7,
        rtol=1e-6,
    )


def test_generator_total_matches_weighted_components() -> None:
    output = make_test_generator_loss_output()

    expected = (
        output.adversarial.weighted
        + output.feature_matching.weighted
        + output.mrstft.weighted
        + output.waveform.weighted
        + output.temporal_gradient.weighted
        + output.rms_envelope.weighted
        + output.band_power.weighted
        + output.spectral_flatness.weighted
        + output.dc_offset.weighted
        + output.cqt_consistency.weighted
        + output.rail_occupancy.weighted
    )

    assert torch.allclose(
        output.total,
        expected,
        atol=1e-7,
        rtol=1e-6,
    )


def test_generator_weighted_components_use_configured_lambdas() -> None:

    output = make_test_generator_loss_output(
        adversarial_weight=0.5,
        feature_matching_weight=1.5,
        mrstft_weight=2.0,
        waveform_weight=0.25,
        temporal_gradient_weight=0.1,
        rms_envelope_weight=0.1,
        band_power_weight=0.1,
        spectral_flatness_weight=0.1,
    )

    assert torch.allclose(
        output.adversarial.weighted,
        0.5 * output.adversarial.raw,
    )

    assert torch.allclose(
        output.feature_matching.weighted,
        1.5 * output.feature_matching.raw,
    )

    assert torch.allclose(
        output.mrstft.weighted,
        2.0 * output.mrstft.raw,
    )

    assert torch.allclose(
        output.waveform.weighted,
        0.25 * output.waveform.raw,
    )

    assert torch.allclose(
        output.temporal_gradient.weighted,
        0.1 * output.temporal_gradient.raw,
    )

    assert torch.allclose(
        output.rms_envelope.weighted,
        0.1 * output.rms_envelope.raw,
    )

    assert torch.allclose(
        output.band_power.weighted,
        0.1 * output.band_power.raw,
    )

    assert torch.allclose(
        output.spectral_flatness.weighted,
        0.1 * output.spectral_flatness.raw,
    )


def test_generator_group_totals() -> None:
    output = make_test_generator_loss_output()

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
            + output.rail_occupancy.weighted
        ),
    )

    assert torch.allclose(
        output.time_frequency_total,
        output.mrstft.weighted + output.cqt_consistency.weighted,
    )

    assert torch.allclose(
        output.time_macro_total,
        output.rms_envelope.weighted + output.dc_offset.weighted,
    )

    assert torch.allclose(
        output.spectral_macro_total,
        output.band_power.weighted + output.spectral_flatness.weighted,
    )


def test_loss_contribution_metrics_sum_to_one() -> None:
    output = make_test_generator_loss_output()

    metrics = (
        generator_loss_contribution_metrics(
            output
        )
    )

    total = sum(metrics.values())

    assert total == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_generator_loss_metrics_use_structured_output() -> None:
    output = make_test_generator_loss_output()

    metrics = generator_loss_metrics(output)

    assert metrics["loss_g_total"] == pytest.approx(
        output.total.item()
    )

    assert metrics[
        "loss_g_adversarial"
    ] == pytest.approx(
        output.adversarial.raw.item()
    )

    assert metrics[
        "loss_g_weighted/adversarial"
    ] == pytest.approx(
        output.adversarial.weighted.item()
    )

    assert metrics[
        "loss_g_group/learned"
    ] == pytest.approx(
        output.learned_total.item()
    )


def test_generator_loss_metrics_include_family_values() -> None:
    output = make_test_generator_loss_output(
        adversarial_raw=2.0,
        adversarial_families={
            "mpd": torch.Tensor((1.0, )),
            "mrd": torch.Tensor((3.0, )),
        },
        feature_matching_raw=3.0,
        feature_matching_families={
            "mpd": torch.Tensor((2.0, )),
            "mrd": torch.Tensor((4.0, )),
        },
    )

    metrics = generator_loss_metrics(output)

    assert metrics[
        "loss_g_adversarial_mpd"
    ] == pytest.approx(1.0)

    assert metrics[
        "loss_g_adversarial_mrd"
    ] == pytest.approx(3.0)

    assert metrics[
        "loss_g_feature_matching_mpd"
    ] == pytest.approx(2.0)

    assert metrics[
        "loss_g_feature_matching_mrd"
    ] == pytest.approx(4.0)


def test_discriminator_loss_metrics_include_family_values() -> None:
    mpd = make_discriminator_family(
        real_output_values=(1.0,),
        fake_output_values=(0.0,),
    )

    mrd = make_discriminator_family(
        real_output_values=(0.0,),
        fake_output_values=(1.0,),
    )

    output = family_balanced_discriminator_loss(
        make_multi_family_discriminator_outputs(
            mpd=mpd,
            mrd=mrd,
        )
    )

    metrics = discriminator_loss_metrics(
        output
    )

    assert metrics[
        "loss_d_mpd_total"
    ] == pytest.approx(0.0)

    assert metrics[
        "loss_d_mrd_total"
    ] == pytest.approx(2.0)

    assert metrics[
        "loss_d_total"
    ] == pytest.approx(1.0)


def test_mean_enabled_families() -> None:
    losses = {
        "mpd": torch.tensor(2.0),
        "mrd": torch.tensor(4.0),
    }

    result = mean_enabled_families(
        losses,
        name="test",
    )

    assert result.item() == pytest.approx(
        3.0
    )


def test_mean_enabled_families_rejects_empty_input() -> None:
    with pytest.raises(
        ValueError,
        match="at least one enabled",
    ):
        mean_enabled_families(
            {},
            name="test",
        )


def test_generator_adversarial_balances_families_equally() -> None:
    mpd = make_discriminator_family(
        real_output_values=(
            1.0,
            1.0,
            1.0,
        ),
        fake_output_values=(
            0.0,
            0.0,
            0.0,
        ),
    )

    mrd = make_discriminator_family(
        real_output_values=(1.0,),
        fake_output_values=(1.0,),
    )

    outputs = (
        make_multi_family_discriminator_outputs(
            mpd=mpd,
            mrd=mrd,
        )
    )

    result = (
        family_balanced_generator_adversarial_loss(
            outputs
        )
    )

    assert_family_loss_output_consistent(
        result
    )

    # MPD fake=0:
    # (1 - 0)^2 = 1
    assert result.families[
         "mpd"
    ].item() == pytest.approx(1.0)

    # MRD fake=1:
    # (1 - 1)^2 = 0
    assert result.families[
        "mrd"
    ].item() == pytest.approx(0.0)

    # Families receive equal weight despite MPD having
    # three branches and MRD only one.
    assert result.total.item() == pytest.approx(
        0.5
    )


def test_generator_adversarial_is_invariant_to_duplicate_equal_branches() -> None:
    one_mpd_branch = make_discriminator_family(
        real_output_values=(1.0,),
        fake_output_values=(0.0,),
    )

    five_mpd_branches = make_discriminator_family(
        real_output_values=(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ),
       fake_output_values=(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    )

    mrd = make_discriminator_family(
        real_output_values=(1.0,),
        fake_output_values=(1.0,),
    )

    one_result = (
        family_balanced_generator_adversarial_loss(
            make_multi_family_discriminator_outputs(
                mpd=one_mpd_branch,
                mrd=mrd,
            )
        )
    )

    five_result = (
        family_balanced_generator_adversarial_loss(
            make_multi_family_discriminator_outputs(
                mpd=five_mpd_branches,
                mrd=mrd,
            )
        )
    )

    assert torch.allclose(
        one_result.total,
        five_result.total,
    )


def test_feature_matching_balances_families_equally() -> None:
    mpd = make_discriminator_family(
        real_output_values=(1.0, 1.0),
        fake_output_values=(0.0, 0.0),
        real_feature_values=(
            (1.0,),
            (1.0,),
        ),
        fake_feature_values=(
            (0.0,),
            (0.0,),
        ),
    )

    mrd = make_discriminator_family(
        real_output_values=(1.0,),
        fake_output_values=(0.0,),
        real_feature_values=(
            (3.0,),
        ),
        fake_feature_values=(
            (0.0,),
        ),
    )

    outputs = (
        make_multi_family_discriminator_outputs(
            mpd=mpd,
            mrd=mrd,
        )
    )

    result = (
        family_balanced_feature_matching_loss(
            outputs,
            normalization="layer_mean",
            scale_floor=0.01,
            eps=1e-8,
        )
    )

    assert_family_loss_output_consistent(
        result
    )

    assert result.families[
        "mpd"
    ].item() == pytest.approx(1.0)

    assert result.families[
        "mrd"
    ].item() == pytest.approx(3.0)

    assert result.total.item() == pytest.approx(
        2.0
    )


def test_family_feature_matching_respects_real_magnitude_normalization() -> None:
    mpd = make_discriminator_family(
        real_output_values=(1.0,),
        fake_output_values=(0.0,),
        real_feature_values=(
            (2.0,),
        ),
        fake_feature_values=(
            (3.0,),
        ),
    )

    mrd = make_discriminator_family(
        real_output_values=(1.0,),
        fake_output_values=(0.0,),
        real_feature_values=(
            (4.0,),
        ),
        fake_feature_values=(
            (6.0,),
        ),
    )

    outputs = (
        make_multi_family_discriminator_outputs(
            mpd=mpd,
            mrd=mrd,
        )
    )

    result = (
        family_balanced_feature_matching_loss(
            outputs,
            normalization="real_magnitude",
            scale_floor=0.01,
            eps=1e-8,
        )
    )

    # MPD: |3 - 2| / 2 = 0.5
    # MRD: |6 - 4| / 4 = 0.5
    assert result.families[
        "mpd"
    ].item() == pytest.approx(0.5)

    assert result.families[
        "mrd"
    ].item() == pytest.approx(0.5)

    assert result.total.item() == pytest.approx(
        0.5
    )


def test_discriminator_loss_balances_families_equally() -> None:
    mpd = make_discriminator_family(
        real_output_values=(
            1.0,
            1.0,
            1.0,
        ),
        fake_output_values=(
            0.0,
            0.0,
            0.0,
        ),
    )

    mrd = make_discriminator_family(
        real_output_values=(0.0,),
        fake_output_values=(1.0,),
    )

    outputs = (
        make_multi_family_discriminator_outputs(
            mpd=mpd,
            mrd=mrd,
        )
    )

    result = (
        family_balanced_discriminator_loss(
            outputs
        )
    )

    assert_family_discriminator_loss_consistent(
        result
    )

    assert result.family_totals[
        "mpd"
    ].item() == pytest.approx(0.0)

    assert result.family_real[
        "mrd"
    ].item() == pytest.approx(1.0)

    assert result.family_fake[
        "mrd"
    ].item() == pytest.approx(1.0)

    assert result.family_totals[
        "mrd"
    ].item() == pytest.approx(2.0)

    assert result.total.item() == pytest.approx(
        1.0
    )


def test_single_enabled_family_is_not_rescaled() -> None:
    mpd = make_discriminator_family(
        real_output_values=(1.0, 0.8),
        fake_output_values=(0.0, 0.2),
    )

    outputs = (
        make_multi_family_discriminator_outputs(
            mpd=mpd,
        )
    )

    generator_result = (
        family_balanced_generator_adversarial_loss(
            outputs
        )
    )

    discriminator_result = (
        family_balanced_discriminator_loss(
            outputs
        )
    )

    assert set(
        generator_result.families
    ) == {"mpd"}

    assert set(
        discriminator_result.family_totals
    ) == {"mpd"}

    assert torch.allclose(
        generator_result.total,
        generator_result.families["mpd"],
    )

    assert torch.allclose(
        discriminator_result.total,
        discriminator_result.family_totals[
            "mpd"
        ],
    )


def test_family_balanced_losses_reject_no_enabled_families() -> None:
    outputs = (
        make_multi_family_discriminator_outputs()
    )

    with pytest.raises(
        ValueError,
        match="at least one enabled",
    ):
        family_balanced_generator_adversarial_loss(
            outputs
        )

    with pytest.raises(
        ValueError,
        match="at least one enabled",
    ):
        family_balanced_feature_matching_loss(
            outputs,
            normalization="layer_mean",
            scale_floor=0.01,
            eps=1e-8,
        )

    with pytest.raises(
        ValueError,
        match="at least one enabled",
    ):
        family_balanced_discriminator_loss(
            outputs
        )


def test_family_balanced_adversarial_loss_backpropagates_to_every_family() -> None:
    mpd = make_discriminator_family(
        real_output_values=(1.0, 1.0),
        fake_output_values=(0.2, 0.3),
        fake_requires_grad=True,
    )

    mrd = make_discriminator_family(
        real_output_values=(1.0,),
        fake_output_values=(0.4,),
        fake_requires_grad=True,
    )

    outputs = (
        make_multi_family_discriminator_outputs(
            mpd=mpd,
            mrd=mrd,
        )
    )

    result = (
        family_balanced_generator_adversarial_loss(
            outputs
        )
    )

    result.total.backward()

    for family in ("mpd", "mrd"):
        for prediction in outputs[
            f"{family}_fake_outputs"
        ]:
            assert prediction.grad is not None
            assert torch.isfinite(
                prediction.grad
            ).all()


def test_rms_envelope_loss_is_zero_for_exact_match() -> None:
    config = RMSEnvelopeLossConfig(
        enabled=True,
        window_sizes=(16, 32, 64),
        hop_ratio=4,
        eps=1e-8,
    )

    loss_function = (
        MultiScaleRMSEnvelopeLoss(config)
    )

    waveform = torch.randn(
        2,
        1,
        256,
    )

    output = loss_function(
        waveform,
        waveform,
    )

    assert output.total.item() == pytest.approx(
        0.0,
        abs=1e-7,
    )


def test_rms_envelope_loss_detects_amplitude_mismatch() -> None:
    config = RMSEnvelopeLossConfig(
        enabled=True,
        window_sizes=(16, 64),
        hop_ratio=4,
        eps=1e-8,
    )

    loss_function = (
        MultiScaleRMSEnvelopeLoss(config)
    )

    target = torch.ones(
        1,
        1,
        256,
    )

    prediction = 0.5 * target

    output = loss_function(
        prediction,
        target,
    )

    assert output.total.item() > 0.0


def test_rms_envelope_is_less_phase_sensitive_than_waveform_l1() -> None:
    time = torch.arange(
        1024,
        dtype=torch.float32,
    )

    target = torch.sin(
        2.0
        * torch.pi
        * time
        / 32.0
    ).reshape(1, 1, -1)

    prediction = torch.roll(
        target,
        shifts=4,
        dims=-1,
    )

    loss_function = MultiScaleRMSEnvelopeLoss(
        RMSEnvelopeLossConfig(
            enabled=True,
            window_sizes=(64, 128),
            hop_ratio=4,
            eps=1e-8,
        )
    )

    envelope_loss = loss_function(
        prediction,
        target,
    ).total

    waveform_loss = F.l1_loss(
        prediction,
        target,
    )

    assert envelope_loss < waveform_loss


def test_rms_envelope_total_is_mean_of_scales() -> None:
    loss_function = MultiScaleRMSEnvelopeLoss(
        RMSEnvelopeLossConfig(
            enabled=True,
            window_sizes=(16, 32, 64),
            hop_ratio=4,
            eps=1e-8,
        )
    )

    prediction = torch.randn(1, 1, 256)
    target = torch.randn(1, 1, 256)

    output = loss_function(
        prediction,
        target,
    )

    expected = torch.stack(
        [
            scale.loss
            for scale in output.scales
        ]
    ).mean()

    assert torch.allclose(
        output.total,
        expected,
    )


def test_rms_envelope_loss_backpropagates() -> None:
    loss_function = MultiScaleRMSEnvelopeLoss(
        RMSEnvelopeLossConfig(
            enabled=True,
            window_sizes=(16, 32, 64),
            hop_ratio=4,
            eps=1e-8,
        )
    )

    prediction = torch.randn(
        2,
        1,
        256,
        requires_grad=True,
    )

    target = torch.randn(
        2,
        1,
        256,
    )

    output = loss_function(
        prediction,
        target,
    )

    output.total.backward()

    assert prediction.grad is not None
    assert torch.isfinite(
        prediction.grad
    ).all()

    assert prediction.grad.abs().sum() > 0


def test_rms_envelope_loss_handles_silence() -> None:
    loss_function = MultiScaleRMSEnvelopeLoss(
        RMSEnvelopeLossConfig(
            enabled=True,
            window_sizes=(16, 64),
            hop_ratio=4,
            eps=1e-8,
        )
    )

    silence = torch.zeros(
        2,
        1,
        256,
    )

    output = loss_function(
        silence,
        silence,
    )

    assert torch.isfinite(output.total)
    assert output.total.item() == pytest.approx(
        0.0,
        abs=1e-7,
    )


def test_log_filterbank_shape() -> None:
    filterbank = (
        build_log_frequency_filterbank(
            sample_rate=44100,
            fft_size=1024,
            number_of_bands=24,
            fmin=27.5,
            fmax=22050.0,
        )
    )

    assert filterbank.shape == (
        24,
        513,
    )

    assert torch.isfinite(
        filterbank
    ).all()

    assert torch.all(
        filterbank >= 0.0
    )


def test_log_filterbank_partitions_covered_bins() -> None:
    filterbank = (
        build_log_frequency_filterbank(
            sample_rate=44100,
            fft_size=1024,
            number_of_bands=24,
            fmin=27.5,
            fmax=22050.0,
        )
    )

    coverage = filterbank.sum(dim=0)
    covered = coverage > 0.0

    assert torch.allclose(
        coverage[covered],
        torch.ones_like(
            coverage[covered]
        ),
        atol=1e-6,
        rtol=1e-6,
    )


def test_band_power_loss_is_zero_for_exact_match() -> None:
    loss_function = (
        MultiResolutionBandPowerLoss(
            sample_rate=8000,
            config=BandPowerLossConfig(
                enabled=True,
                fft_sizes=(64, 128),
                hop_sizes=(16, 32),
                win_lengths=(64, 128),
                number_of_bands=8,
                fmin=50.0,
                fmax=4000.0,
                absolute_weight=1.0,
                shape_weight=1.0,
                eps=1e-8,
            ),
        )
    )

    waveform = torch.randn(
        2,
        1,
        512,
    )

    output = loss_function(
        waveform,
        waveform,
    )

    assert output.total.item() == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_band_power_separates_absolute_energy_and_shape() -> None:
    loss_function = (
        MultiResolutionBandPowerLoss(
            sample_rate=8000,
            config=BandPowerLossConfig(
                enabled=True,
                fft_sizes=(128,),
                hop_sizes=(32,),
                win_lengths=(128,),
                number_of_bands=8,
                fmin=50.0,
                fmax=4000.0,
                absolute_weight=1.0,
                shape_weight=1.0,
                eps=1e-8,
            ),
        )
    )

    target = torch.randn(
        2,
        1,
        512,
    )

    prediction = 0.5 * target

    output = loss_function(
        prediction,
        target,
    )

    assert output.absolute.item() > 0.0

    assert output.shape.item() == pytest.approx(
        0.0,
        abs=1e-5,
    )


def test_band_power_shape_detects_frequency_shift() -> None:
    sample_rate = 8000
    length = 1024

    time = torch.arange(
        length,
        dtype=torch.float32,
    ) / sample_rate

    target = torch.sin(
        2.0 * torch.pi * 220.0 * time
    ).reshape(1, 1, -1)

    prediction = torch.sin(
        2.0 * torch.pi * 880.0 * time
    ).reshape(1, 1, -1)

    loss_function = (
        MultiResolutionBandPowerLoss(
            sample_rate=sample_rate,
            config=BandPowerLossConfig(
                enabled=True,
                fft_sizes=(256,),
                hop_sizes=(64,),
                win_lengths=(256,),
                number_of_bands=12,
                fmin=40.0,
                fmax=4000.0,
                absolute_weight=1.0,
                shape_weight=1.0,
                eps=1e-8,
            ),
        )
    )

    output = loss_function(
        prediction,
        target,
    )

    assert output.shape.item() > 0.0


def test_band_power_total_matches_components() -> None:
    config = BandPowerLossConfig(
        enabled=True,
        fft_sizes=(64, 128),
        hop_sizes=(16, 32),
        win_lengths=(64, 128),
        number_of_bands=8,
        fmin=50.0,
        fmax=4000.0,
        absolute_weight=2.0,
        shape_weight=0.5,
        eps=1e-8,
    )

    loss_function = (
        MultiResolutionBandPowerLoss(
            sample_rate=8000,
            config=config,
        )
    )

    prediction = torch.randn(1, 1, 512)
    target = torch.randn(1, 1, 512)

    output = loss_function(
        prediction,
        target,
    )

    expected = (
        config.absolute_weight
        * output.absolute
        + config.shape_weight
        * output.shape
    )

    assert torch.allclose(
        output.total,
        expected,
        atol=1e-7,
        rtol=1e-6,
    )


def test_band_power_loss_backpropagates() -> None:
    loss_function = (
        MultiResolutionBandPowerLoss(
            sample_rate=8000,
            config=BandPowerLossConfig(
                enabled=True,
                fft_sizes=(64, 128),
                hop_sizes=(16, 32),
                win_lengths=(64, 128),
                number_of_bands=8,
                fmin=50.0,
                fmax=4000.0,
                absolute_weight=1.0,
                shape_weight=1.0,
                eps=1e-8,
            ),
        )
    )

    prediction = torch.randn(
        2,
        1,
        512,
        requires_grad=True,
    )

    target = torch.randn(
        2,
        1,
        512,
    )

    output = loss_function(
        prediction,
        target,
    )

    output.total.backward()

    assert prediction.grad is not None

    assert torch.isfinite(
        prediction.grad
    ).all()

    assert prediction.grad.abs().sum() > 0


def test_band_power_loss_handles_silence() -> None:
    loss_function = (
        MultiResolutionBandPowerLoss(
            sample_rate=8000,
            config=BandPowerLossConfig(
                enabled=True,
                fft_sizes=(64,),
                hop_sizes=(16,),
                win_lengths=(64,),
                number_of_bands=8,
                fmin=50.0,
                fmax=4000.0,
                absolute_weight=1.0,
                shape_weight=1.0,
                eps=1e-8,
            ),
        )
    )

    silence = torch.zeros(
        2,
        1,
        512,
    )

    output = loss_function(
        silence,
        silence,
    )

    assert torch.isfinite(output.total)

    assert output.total.item() == pytest.approx(
        0.0,
        abs=1e-7,
    )


@pytest.mark.parametrize(
    "fft_size",
    [1024, 4096],
)
def test_log_filterbank_contains_no_empty_bands(
    fft_size: int,
) -> None:
    filterbank = (
        build_log_frequency_filterbank(
            sample_rate=44100,
            fft_size=fft_size,
            number_of_bands=24,
            fmin=27.5,
            fmax=22050.0,
        )
    )

    assert torch.all(
        filterbank.sum(dim=1) > 0.0
    )


def test_adaptive_filterbank_partitions_covered_bins() -> None:
    filterbank = (
        build_log_frequency_filterbank(
            sample_rate=44100,
            fft_size=1024,
            number_of_bands=24,
            fmin=27.5,
            fmax=22050.0,
        )
    )

    coverage = filterbank.sum(dim=0)
    covered = coverage > 0.0

    assert torch.allclose(
        coverage[covered],
        torch.ones_like(
            coverage[covered]
        ),
        atol=1e-6,
        rtol=1e-6,
    )


def test_spectral_convergence_averages_per_example() -> None:
    fake = torch.tensor(
        [
            [[2.0, 0.0]],
            [[4.0, 0.0]],
        ]
    )

    real = torch.tensor(
        [
            [[1.0, 0.0]],
            [[2.0, 0.0]],
        ]
    )

    actual, _, _, _ = (
        SingleResolutionSTFTLoss
        ._stft_spectral_convergence(
            fake,
            real,
            reference_floor=1.0,
        )
    )

    first = torch.linalg.vector_norm(
        real[0] - fake[0]
    ) / torch.linalg.vector_norm(
        real[0]
    ).clamp_min(1.0)

    second = torch.linalg.vector_norm(
        real[1] - fake[1]
    ) / torch.linalg.vector_norm(
        real[1]
    ).clamp_min(1.0)

    expected = 0.5 * (
        first + second
    )

    assert torch.allclose(
        actual,
        expected,
    )


def test_spectral_convergence_is_stable_for_silent_target() -> None:
    fake = torch.ones(
        2,
        16,
        8,
        requires_grad=True,
    )

    real = torch.zeros_like(fake)

    loss, _, _, _ = (
        SingleResolutionSTFTLoss
        ._stft_spectral_convergence(
            fake,
            real,
            reference_floor=1.0,
        )
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert fake.grad is not None
    assert torch.isfinite(fake.grad).all()

    assert loss.item() < 1e4


def test_spectral_convergence_ignores_fully_silent_reference() -> None:
    fake = torch.ones(
        2,
        16,
        8,
        requires_grad=True,
    )

    real = torch.zeros_like(fake)

    loss, _, _, _ = (
        SingleResolutionSTFTLoss
        ._stft_spectral_convergence(
            fake,
            real,
            reference_floor=1.0,
        )
    )

    loss.backward()

    assert loss.item() == pytest.approx(
        0.0,
        abs=1e-7,
    )

    assert fake.grad is not None
    assert torch.isfinite(fake.grad).all()


def test_spectral_convergence_matches_standard_formula_for_active_target() -> None:
    real = torch.full(
        (2, 16, 8),
        2.0,
    )

    fake = real + 0.25

    actual, _, _, _ = (
        SingleResolutionSTFTLoss
        ._stft_spectral_convergence(
            fake,
            real,
            reference_floor=1.0,
        )
    )

    difference_norm = torch.linalg.vector_norm(
        real - fake,
        dim=(-2, -1),
    )

    reference_norm = torch.linalg.vector_norm(
        real,
        dim=(-2, -1),
    )

    expected = (
        difference_norm
        / reference_norm
    ).mean()

    assert torch.allclose(
        actual,
        expected,
        atol=1e-7,
        rtol=1e-6,
    )


def test_spectral_convergence_downweights_quiet_reference() -> None:
    real = torch.zeros(
        2,
        16,
        8,
    )

    real[0].fill_(1.0)
    real[1].fill_(0.001)

    fake = real + 0.1

    loss, _, _, _ = (
        SingleResolutionSTFTLoss
        ._stft_spectral_convergence(
            fake,
            real,
            reference_floor=1.0,
        )
    )

    assert torch.isfinite(loss)
    assert loss.item() < 10.0


def test_silent_example_does_not_dominate_active_example() -> None:
    real = torch.zeros(
        2,
        16,
        8,
    )

    real[0].fill_(1.0)

    fake = real.clone()
    fake[0] += 0.1
    fake[1] += 10.0

    mixed_loss, _, _, _ = (
        SingleResolutionSTFTLoss
        ._stft_spectral_convergence(
            fake,
            real,
            reference_floor=1.0,
        )
    )

    active_only_loss, _, _, _ = (
        SingleResolutionSTFTLoss
        ._stft_spectral_convergence(
            fake[:1],
            real[:1],
            reference_floor=1.0,
        )
    )

    assert torch.allclose(
        mixed_loss,
        active_only_loss,
        atol=1e-6,
        rtol=1e-5,
    )


def test_log_spectral_flatness_is_zero_for_flat_power() -> None:
    power = torch.ones(
        2,
        1,
        16,
        8,
    )

    flatness = log_spectral_flatness(
        power,
        frequency_dim=-2,
        eps=1e-8,
    )

    assert torch.allclose(
        flatness,
        torch.zeros_like(flatness),
        atol=1e-7,
    )


def test_log_spectral_flatness_detects_tonal_power() -> None:
    flat = torch.ones(
        1,
        1,
        16,
        1,
    )

    tonal = torch.full_like(
        flat,
        1e-4,
    )

    tonal[..., 4, :] = 1.0

    flat_value = log_spectral_flatness(
        flat,
        frequency_dim=-2,
        eps=1e-8,
    )

    tonal_value = log_spectral_flatness(
        tonal,
        frequency_dim=-2,
        eps=1e-8,
    )

    assert torch.all(
        tonal_value < flat_value
    )


def test_spectral_flatness_loss_is_zero_for_exact_match() -> None:
    loss_function = (
        SubBandSpectralFlatnessLoss(
            sample_rate=8000,
            config=(
                SpectralFlatnessLossConfig(
                    enabled=True,
                    fft_size=128,
                    hop_size=32,
                    win_length=128,
                    band_edges=(
                        0.0,
                        500.0,
                        2000.0,
                        4000.0,
                    ),
                    energy_floor=1e-4,
                    minimum_bins_per_band=4,
                    eps=1e-8,
                )
            ),
        )
    )

    waveform = torch.randn(
        2,
        1,
        1024,
    )

    output = loss_function(
        waveform,
        waveform,
    )

    assert output.total.item() == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_spectral_flatness_distinguishes_tone_and_noise() -> None:
    sample_rate = 8000
    length = 2048

    time = torch.arange(
        length,
        dtype=torch.float32,
    ) / sample_rate

    tone = torch.sin(
        2.0
        * torch.pi
        * 1000.0
        * time
    ).reshape(1, 1, -1)

    generator = torch.Generator()
    generator.manual_seed(1337)

    noise = torch.randn(
        1,
        1,
        length,
        generator=generator,
    )

    noise = noise / noise.square().mean().sqrt()

    loss_function = (
        SubBandSpectralFlatnessLoss(
            sample_rate=sample_rate,
            config=(
                SpectralFlatnessLossConfig(
                    enabled=True,
                    fft_size=256,
                    hop_size=64,
                    win_length=256,
                    band_edges=(
                        0.0,
                        500.0,
                        2000.0,
                        4000.0,
                    ),
                    energy_floor=1e-4,
                    minimum_bins_per_band=4,
                    eps=1e-8,
                )
            ),
        )
    )

    output = loss_function(
        tone,
        noise,
    )

    assert output.total.item() > 0.0


def test_spectral_flatness_is_scale_invariant_for_active_audio() -> None:
    loss_function = (
        make_test_spectral_flatness_loss()
    )

    target = torch.randn(
        2,
        1,
        2048,
    )

    prediction = torch.roll(
        target,
        shifts=3,
        dims=-1,
    )

    first = loss_function(
        prediction,
        target,
    ).total

    second = loss_function(
        0.5 * prediction,
        0.5 * target,
    ).total

    assert torch.allclose(
        first,
        second,
        atol=1e-3,
        rtol=1e-3,
    )



def test_spectral_flatness_loss_handles_silence() -> None:
    loss_function = (
        make_test_spectral_flatness_loss()
    )

    target = torch.zeros(
        2,
        1,
        2048,
    )

    prediction = torch.randn_like(
        target
    )

    output = loss_function(
        prediction,
        target,
    )

    assert torch.isfinite(output.total)

    assert output.total.item() == pytest.approx(
        0.0,
        abs=1e-7,
    )

    for band in output.bands:
        assert (
            band.activity_fraction.item()
            == pytest.approx(
                0.0,
                abs=1e-7,
            )
        )



def test_spectral_flatness_loss_backpropagates() -> None:
    loss_function = (
        make_test_spectral_flatness_loss()
    )

    prediction = torch.randn(
        2,
        1,
        2048,
        requires_grad=True,
    )

    target = torch.randn(
        2,
        1,
        2048,
    )

    output = loss_function(
        prediction,
        target,
    )

    output.total.backward()

    assert prediction.grad is not None

    assert torch.isfinite(
        prediction.grad
    ).all()

    assert prediction.grad.abs().sum() > 0



def test_spectral_flatness_rejects_bands_with_too_few_bins() -> None:
    with pytest.raises(
        ValueError,
        match="contains only",
    ):
        SubBandSpectralFlatnessLoss(
            sample_rate=44100,
            config=(
                SpectralFlatnessLossConfig(
                    enabled=True,
                    fft_size=128,
                    hop_size=32,
                    win_length=128,
                    band_edges=(
                        27.5,
                        50.0,
                        22050.0,
                    ),
                    minimum_bins_per_band=4,
                )
            ),
        )


def test_slew_envelope_is_zero_for_exact_match() -> None:
    loss_function = (
        MultiScaleSlewEnvelopeLoss(
            lags=(1, 2, 4),
            window_sizes=(16, 64),
            hop_ratio=4,
            mean_weight=1.0,
            peak_weight=0.5,
        )
    )

    waveform = torch.randn(
        2,
        1,
        512,
    )

    output = loss_function(
        waveform,
        waveform,
    )

    assert output.total.item() == pytest.approx(
        0.0,
        abs=1e-7,
    )


def test_slew_envelope_is_polarity_invariant() -> None:
    loss_function = (
        MultiScaleSlewEnvelopeLoss(
            lags=(1, 2, 4),
            window_sizes=(16, 64),
        )
    )

    waveform = torch.randn(
        2,
        1,
        512,
    )

    output = loss_function(
        -waveform,
        waveform,
    )

    assert output.total.item() == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_peak_slew_penalizes_smoothed_transition() -> None:
    target = torch.full(
        (1, 1, 256),
        -1.0,
    )

    target[..., 128:] = 1.0

    prediction = target.clone()

    prediction[
        ...,
        120:136,
    ] = torch.linspace(
        -1.0,
        1.0,
        16,
    )

    loss_function = (
        MultiScaleSlewEnvelopeLoss(
            lags=(1,),
            window_sizes=(32,),
            hop_ratio=4,
            mean_weight=0.0,
            peak_weight=1.0,
        )
    )

    output = loss_function(
        prediction,
        target,
    )

    assert output.peak.item() > 0.0


def test_mean_slew_detects_missing_edges() -> None:
    target = torch.sign(
        torch.sin(
            2.0
            * torch.pi
            * torch.arange(512)
            / 32.0
        )
    ).reshape(1, 1, -1)

    prediction = torch.zeros_like(
        target
    )

    loss_function = (
        MultiScaleSlewEnvelopeLoss(
            lags=(1, 2),
            window_sizes=(32, 64),
            mean_weight=1.0,
            peak_weight=0.0,
        )
    )

    output = loss_function(
        prediction,
        target,
    )

    assert output.mean.item() > 0.0


def test_slew_envelope_penalizes_chatter_in_silence() -> None:
    target = torch.zeros(
        1,
        1,
        512,
    )

    prediction = torch.ones_like(
        target
    )

    prediction[..., 1::2] = -1.0

    loss_function = (
        MultiScaleSlewEnvelopeLoss(
            lags=(1, 2, 4),
            window_sizes=(16, 64),
        )
    )

    output = loss_function(
        prediction,
        target,
    )

    assert output.total.item() > 0.0


def test_slew_envelope_backpropagates() -> None:
    loss_function = (
        MultiScaleSlewEnvelopeLoss(
            lags=(1, 2, 4),
            window_sizes=(16, 64),
        )
    )

    prediction = torch.randn(
        2,
        1,
        512,
        requires_grad=True,
    )

    target = torch.randn(
        2,
        1,
        512,
    )

    output = loss_function(
        prediction,
        target,
    )

    output.total.backward()

    assert prediction.grad is not None

    assert torch.isfinite(
        prediction.grad
    ).all()

    assert prediction.grad.abs().sum() > 0


def test_slew_envelope_relaxes_edge_position_within_pooling_window() -> None:
    target = torch.zeros(
        1,
        1,
        256,
    )

    # Target transition occurs between samples 99 and 100.
    target[..., 100:] = 1.0

    prediction = torch.zeros_like(
        target
    )

    # Generated transition is shifted by four samples.
    prediction[..., 104:] = 1.0

    pointwise_loss = (
        MultiScaleTemporalGradientLoss(
            lags=(1,),
        )(
            prediction,
            target,
        )
    )

    relaxed_output = (
        MultiScaleSlewEnvelopeLoss(
            lags=(1,),
            window_sizes=(32,),
            hop_ratio=1,
            mean_weight=1.0,
            peak_weight=0.5,
        )(
            prediction,
            target,
        )
    )

    # Both transitions remain inside the same 32-sample
    # pooling region: samples 96–127.
    #
    # Both have:
    # - the same total slew,
    # - the same peak slew,
    # - the same pooled location.
    assert relaxed_output.mean.item() == pytest.approx(
        0.0,
        abs=1e-7,
    )

    assert relaxed_output.peak.item() == pytest.approx(
        0.0,
        abs=1e-7,
    )

    assert relaxed_output.total.item() == pytest.approx(
        0.0,
        abs=1e-7,
    )

    # Pointwise derivative matching still detects the
    # exact sample-position difference.
    assert pointwise_loss.item() > 0.0


def test_overlapping_slew_envelope_prefers_small_shift_over_missing_edge() -> None:
    target = torch.zeros(
        1,
        1,
        512,
    )

    target[..., 128:] = 1.0

    slightly_shifted = torch.zeros_like(
        target
    )

    slightly_shifted[..., 132:] = 1.0

    missing = torch.zeros_like(
        target
    )

    loss_function = (
        MultiScaleSlewEnvelopeLoss(
            lags=(1,),
            window_sizes=(32, 64),
            hop_ratio=4,
            mean_weight=1.0,
            peak_weight=0.5,
        )
    )

    shifted_loss = loss_function(
        slightly_shifted,
        target,
    ).total

    missing_loss = loss_function(
        missing,
        target,
    ).total

    assert shifted_loss < missing_loss


def test_slew_envelope_penalty_grows_when_edge_moves_outside_local_region() -> None:
    target = torch.zeros(
        1,
        1,
        512,
    )

    target[..., 128:] = 1.0

    near = torch.zeros_like(
        target
    )
    near[..., 132:] = 1.0

    far = torch.zeros_like(
        target
    )
    far[..., 224:] = 1.0

    loss_function = (
        MultiScaleSlewEnvelopeLoss(
            lags=(1,),
            window_sizes=(32, 64),
            hop_ratio=4,
            mean_weight=1.0,
            peak_weight=0.5,
        )
    )

    near_loss = loss_function(
        near,
        target,
    ).total

    far_loss = loss_function(
        far,
        target,
    ).total

    assert near_loss < far_loss


def test_dc_offset_loss_is_zero_for_exact_match() -> None:
    loss_function = MultiScaleDCOffsetLoss(
        DCOffsetLossConfig(
            enabled=True,
            window_sizes=(16, 64, 128),
            hop_ratio=4,
        )
    )

    waveform = torch.randn(
        2,
        1,
        512,
    )

    output = loss_function(
        waveform,
        waveform,
    )

    assert output.total.item() == pytest.approx(
        0.0,
        abs=1e-7,
    )


def test_dc_offset_loss_detects_constant_offset() -> None:
    loss_function = MultiScaleDCOffsetLoss(
        DCOffsetLossConfig(
            enabled=True,
            window_sizes=(16, 64, 128),
            hop_ratio=4,
        )
    )

    target = torch.zeros(
        2,
        1,
        512,
    )

    prediction = target + 0.25

    output = loss_function(
        prediction,
        target,
    )

    assert output.local.item() == pytest.approx(
        0.25,
        abs=1e-6,
    )

    assert (
        output.global_mean.item()
        == pytest.approx(
            0.25,
            abs=1e-6,
        )
    )

    assert output.total.item() == pytest.approx(
        0.25,
        abs=1e-6,
    )


def test_dc_offset_loss_detects_local_offsets_that_cancel_globally() -> None:
    loss_function = MultiScaleDCOffsetLoss(
        DCOffsetLossConfig(
            enabled=True,
            window_sizes=(32, 64),
            hop_ratio=4,
        )
    )

    target = torch.zeros(
        1,
        1,
        512,
    )

    prediction = torch.zeros_like(
        target
    )

    prediction[..., :256] = 0.25
    prediction[..., 256:] = -0.25

    output = loss_function(
        prediction,
        target,
    )

    assert (
        output.global_mean.item()
        == pytest.approx(
            0.0,
            abs=1e-7,
        )
    )

    assert output.local.item() > 0.0
    assert output.total.item() > 0.0


def test_dc_offset_loss_matches_target_mean_instead_of_zero() -> None:
    loss_function = MultiScaleDCOffsetLoss(
        DCOffsetLossConfig(
            enabled=True,
            window_sizes=(32, 64),
            hop_ratio=4,
        )
    )

    target = torch.full(
        (1, 1, 512),
        0.2,
    )

    matching = target.clone()

    zero_centered = torch.zeros_like(
        target
    )

    matching_loss = loss_function(
        matching,
        target,
    ).total

    zero_loss = loss_function(
        zero_centered,
        target,
    ).total

    assert matching_loss.item() == pytest.approx(
        0.0,
        abs=1e-7,
    )

    assert zero_loss.item() > 0.0


def test_dc_offset_loss_is_phase_relaxed_for_zero_mean_periodic_signal() -> None:
    time = torch.arange(
        1024,
        dtype=torch.float32,
    )

    target = torch.sin(
        2.0
        * torch.pi
        * time
        / 32.0
    ).reshape(1, 1, -1)

    prediction = torch.roll(
        target,
        shifts=8,
        dims=-1,
    )

    loss_function = MultiScaleDCOffsetLoss(
        DCOffsetLossConfig(
            enabled=True,
            # Integer multiples of the period.
            window_sizes=(32, 64, 128),
            hop_ratio=4,
        )
    )

    output = loss_function(
        prediction,
        target,
    )

    assert output.total.item() == pytest.approx(
        0.0,
        abs=1e-5,
    )


def test_dc_offset_loss_backpropagates() -> None:
    loss_function = MultiScaleDCOffsetLoss(
        DCOffsetLossConfig(
            enabled=True,
            window_sizes=(16, 64),
            hop_ratio=4,
        )
    )

    prediction = torch.randn(
        2,
        1,
        512,
        requires_grad=True,
    )

    target = torch.randn(
        2,
        1,
        512,
    )

    output = loss_function(
        prediction,
        target,
    )

    output.total.backward()

    assert prediction.grad is not None

    assert torch.isfinite(
        prediction.grad
    ).all()

    assert prediction.grad.abs().sum() > 0


def test_cqt_loss_is_zero_for_exact_match() -> None:
    loss_function = make_test_cqt_loss()

    waveform = torch.randn(
        2,
        1,
        4096,
    )

    output = loss_function(
        waveform,
        waveform,
    )

    assert output.total.item() == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_cqt_loss_detects_amplitude_mismatch() -> None:
    loss_function = make_test_cqt_loss()

    target = torch.randn(
        2,
        1,
        4096,
    )

    prediction = 0.5 * target

    output = loss_function(
        prediction,
        target,
    )

    assert output.total.item() > 0.0


def test_cqt_loss_is_loss_phase_sensitive_than_waveform_l1() -> None:
    loss_function = make_test_cqt_loss()

    time = torch.arange(
        4096,
        dtype=torch.float32,
    )

    target = torch.sin(2.0*torch.pi * 440.0*time / 8_000.0).reshape(1, 1, -1)

    prediction = torch.roll(
        target,
        shifts=4,
        dims=-1,
    )

    cqt_loss = loss_function(
        prediction,
        target,
    ).total

    waveform_loss = F.l1_loss(
        prediction,
        target,
    )

    assert cqt_loss < waveform_loss


def test_cqt_loss_backpropagates_only_to_prediction() -> None:
    loss_function = make_test_cqt_loss()

    prediction = torch.randn(
        2,
        1,
        4096,
        requires_grad=True,
    )

    target = torch.randn(
        2,
        1,
        4096,
        requires_grad=True,
    )

    output = loss_function(
        prediction,
        target,
    )

    output.total.backward()

    assert prediction.grad is not None
    assert torch.isfinite(
        prediction.grad
    ).all()

    assert prediction.grad.abs().sum() > 0

    assert target.grad is None


def test_cqt_loss_has_no_trainable_parameters() -> None:
    loss_function = make_test_cqt_loss()

    trainable = [
        parameter
        for parameter
        in loss_function.parameters()
        if parameter.requires_grad
    ]

    assert trainable == []


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required.",
)
def test_cqt_loss_backpropagates_on_cuda() -> None:
    loss_function = make_test_cqt_loss(
        device="cuda"
    )

    prediction = torch.randn(
        2,
        1,
        4096,
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )

    target = torch.randn_like(
        prediction
    )

    with torch.amp.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):
        output = loss_function(
            prediction,
            target,
        )

    output.total.backward()

    assert output.total.dtype == torch.float32
    assert prediction.grad is not None
    assert torch.isfinite(
        prediction.grad
    ).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required.",
)
def test_cqt_loss_moves_internal_transform_to_cuda() -> None:
    loss_function = make_test_cqt_loss(
        device="cuda"
    )

    prediction = torch.randn(
        1,
        1,
        4096,
        device="cuda",
    )

    output = loss_function(
        prediction,
        prediction,
    )

    assert output.total.is_cuda


def test_rail_occupancy_is_zero_for_exact_match() -> None:
    loss_function = (
        make_test_rail_occupancy_loss()
    )

    waveform = torch.randn(
        2,
        1,
        512,
    ).tanh()

    output = loss_function(
        waveform,
        waveform,
    )

    assert output.total.item() == pytest.approx(
        0.0,
        abs=1e-7,
    )


def test_rail_occupancy_is_polarity_invariant() -> None:
    loss_function = (
        make_test_rail_occupancy_loss()
    )

    target = torch.ones(
        1,
        1,
        512,
    )

    target[..., 1::2] = -1.0

    output = loss_function(
        -target,
        target,
    )

    assert output.total.item() == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_rail_occupancy_penalizes_under_amplitude_pulse() -> None:
    loss_function = (
        make_test_rail_occupancy_loss()
    )

    target = torch.ones(
        1,
        1,
        512,
    )

    target[..., 1::2] = -1.0

    prediction = 0.25 * target

    output = loss_function(
        prediction,
        target,
    )

    assert output.total.item() > 0.0

    assert (
        output.prediction_occupancy
        < output.target_occupancy
    )


def triangle_wave(
    *,
    length: int,
    period: int,
) -> torch.Tensor:
    phase = (
        torch.arange(
            length,
            dtype=torch.float32,
        )
        % period
    ) / float(period)

    waveform = (
        4.0
        * torch.abs(
            phase - 0.5
        )
        - 1.0
    )

    return waveform.reshape(
        1,
        1,
        -1,
    )


def test_square_has_greater_rail_occupancy_than_triangle() -> None:
    loss_function = (
        make_test_rail_occupancy_loss(
            window_sizes=(64,),
            hop_ratio=1,
        )
    )

    square = torch.ones(
        1,
        1,
        512,
    )

    square[..., 32::64] = -1.0

    # Prefer constructing a proper periodic square helper.
    square = torch.sign(
        torch.sin(
            2.0
            * torch.pi
            * torch.arange(512)
            / 64.0
        )
    ).reshape(1, 1, -1)

    triangle = triangle_wave(
        length=512,
        period=64,
    )

    square_output = loss_function(
        square,
        square,
    )

    triangle_output = loss_function(
        triangle,
        triangle,
    )

    assert (
        square_output.target_occupancy
        > triangle_output.target_occupancy
    )


def test_rail_occupancy_ignores_silent_target() -> None:
    loss_function = (
        make_test_rail_occupancy_loss()
    )

    target = torch.zeros(
        2,
        1,
        512,
    )

    prediction = torch.randn_like(
        target
    )

    output = loss_function(
        prediction,
        target,
    )

    assert torch.isfinite(output.total)

    assert output.total.item() == pytest.approx(
        0.0,
        abs=1e-7,
    )

    for scale in output.scales:
        assert (
            scale.activity_fraction.item()
            == pytest.approx(
                0.0,
                abs=1e-7,
            )
        )


def test_rail_occupancy_is_phase_relaxed() -> None:
    loss_function = (
        make_test_rail_occupancy_loss(
            window_sizes=(64, 128),
            hop_ratio=1,
        )
    )

    target = torch.ones(
        1,
        1,
        512,
    )

    target[..., 16::32] = -1.0

    prediction = torch.roll(
        target,
        shifts=4,
        dims=-1,
    )

    output = loss_function(
        prediction,
        target,
    )

    assert output.total.item() == pytest.approx(
        0.0,
        abs=1e-5,
    )


def test_rail_occupancy_backpropagates() -> None:
    loss_function = (
        make_test_rail_occupancy_loss()
    )

    prediction_logits = torch.randn(
        2,
        1,
        512,
        requires_grad=True,
    )

    prediction = torch.tanh(
        prediction_logits
    )

    target = torch.tanh(
        torch.randn(
            2,
            1,
            512,
        )
    )

    output = loss_function(
        prediction,
        target,
    )

    output.total.backward()

    assert prediction_logits.grad is not None

    assert torch.isfinite(
        prediction_logits.grad
    ).all()

    assert (
        prediction_logits
        .grad
        .abs()
        .sum()
        .item()
        > 0.0
    )

