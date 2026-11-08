import pytest
import torch

from vocoder.src.nes_vocoder.training.accumulation import accumulation_windows
from vocoder.src.nes_vocoder.training.runtime import derive_training_seed, set_seed
from vocoder.src.nes_vocoder.training.step import vocoder_train_step, zero_vocoder_gradients, vocoder_train_micro_step, \
    finish_vocoder_optimizer_step
from vocoder.tests.helpers.vocoder_training_helpers import make_training_fixture, assert_module_parameters_equal, \
    clone_parameter_state, assert_module_parameters_changed


def test_accumulation_windows_full_groups():
    values = list(range(8))

    windows = list(
        accumulation_windows(values, 4)
    )

    assert windows == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]


def test_accumulation_windows_partial_final_group():
    values = list(range(6))

    windows = list(
        accumulation_windows(values, 4)
    )

    assert windows == [
        [0, 1, 2, 3],
        [4, 5],
    ]


def test_gradient_accumulation_matches_full_batch():
    torch.manual_seed(1337)

    full_model = torch.nn.Linear(4, 1, bias=False)
    accumulated_model = torch.nn.Linear(
        4,
        1,
        bias=False,
    )

    accumulated_model.load_state_dict(
        full_model.state_dict()
    )

    full_optimizer = torch.optim.SGD(
        full_model.parameters(),
        lr=0.1,
    )
    accumulated_optimizer = torch.optim.SGD(
        accumulated_model.parameters(),
        lr=0.1,
    )

    x = torch.randn(8, 4)
    target = torch.randn(8, 1)

    full_optimizer.zero_grad(set_to_none=True)

    full_prediction = full_model(x)
    full_loss = torch.nn.functional.mse_loss(
        full_prediction,
        target,
    )
    full_loss.backward()
    full_optimizer.step()

    accumulated_optimizer.zero_grad(set_to_none=True)

    microbatches = [
        (x[:4], target[:4]),
        (x[4:], target[4:]),
    ]

    for micro_x, micro_target in microbatches:
        prediction = accumulated_model(micro_x)
        loss = torch.nn.functional.mse_loss(
            prediction,
            micro_target,
        )
        (loss / len(microbatches)).backward()

    accumulated_optimizer.step()

    for full_parameter, accumulated_parameter in zip(
        full_model.parameters(),
        accumulated_model.parameters(),
    ):
        assert torch.allclose(
            full_parameter,
            accumulated_parameter,
            atol=1e-6,
            rtol=1e-5,
        )


def test_train_step_wrapper_matches_canonical_path() -> None:
    wrapped = make_training_fixture(
        seed=1337
    )

    canonical = make_training_fixture(
        seed=1337
    )

    wrapped_result = vocoder_train_step(
        batch=wrapped.batch,
        models=wrapped.models,
        optimizers=wrapped.optimizers,
        loss_bundle=wrapped.loss_bundle,
        device=wrapped.device,
        use_amp=False,
        scaler=None,
        dtype="float16",
        grad_clip_generator=10.0,
        grad_clip_discriminator=10.0,
    )

    zero_vocoder_gradients(
        canonical.optimizers
    )

    micro_result = vocoder_train_micro_step(
        batch=canonical.batch,
        models=canonical.models,
        loss_bundle=canonical.loss_bundle,
        device=canonical.device,
        use_amp=False,
        scaler=None,
        dtype="float16",
        loss_divisor=1,
    )

    gradient_metrics = (
        finish_vocoder_optimizer_step(
            models=canonical.models,
            optimizers=canonical.optimizers,
            scaler=None,
            use_amp=False,
            grad_clip_generator=10.0,
            grad_clip_discriminator=10.0,
        )
    )

    for key, expected in (
        micro_result.losses.items()
    ):
        assert key in wrapped_result.losses

        assert wrapped_result.losses[
            key
        ] == pytest.approx(
            expected,
            rel=1e-5,
            abs=1e-6,
        )

    for key, expected in (
        gradient_metrics.items()
    ):
        if expected is None:
            continue

        assert wrapped_result.losses[
            key
        ] == pytest.approx(
            expected,
            rel=1e-5,
            abs=1e-6,
        )

    assert_module_parameters_equal(
        wrapped.models.generator,
        canonical.models.generator,
    )

    assert_module_parameters_equal(
        wrapped.models.discriminator,
        canonical.models.discriminator,
    )


def test_train_step_wrapper_updates_both_models() -> None:
    fixture = make_training_fixture()

    generator_before = clone_parameter_state(
        fixture.models.generator
    )

    discriminator_before = (
        clone_parameter_state(
            fixture.models.discriminator
        )
    )

    vocoder_train_step(
        batch=fixture.batch,
        models=fixture.models,
        optimizers=fixture.optimizers,
        loss_bundle=fixture.loss_bundle,
        device=fixture.device,
        use_amp=False,
        scaler=None,
        dtype="float16",
        grad_clip_generator=10.0,
        grad_clip_discriminator=10.0,
    )

    assert_module_parameters_changed(
        fixture.models.generator,
        generator_before,
    )

    assert_module_parameters_changed(
        fixture.models.discriminator,
        discriminator_before,
    )


def test_train_step_wrapper_clears_stale_gradients() -> None:
    clean = make_training_fixture(
        seed=1337
    )

    dirty = make_training_fixture(
        seed=1337
    )

    for parameter in (
        dirty.models.generator.parameters()
    ):
        parameter.grad = torch.ones_like(
            parameter
        )

    for parameter in (
        dirty.models.discriminator.parameters()
    ):
        parameter.grad = torch.ones_like(
            parameter
        )

    vocoder_train_step(
        batch=clean.batch,
        models=clean.models,
        optimizers=clean.optimizers,
        loss_bundle=clean.loss_bundle,
        device=clean.device,
        use_amp=False,
        scaler=None,
        dtype="float16",
        grad_clip_generator=10.0,
        grad_clip_discriminator=10.0,
    )

    vocoder_train_step(
        batch=dirty.batch,
        models=dirty.models,
        optimizers=dirty.optimizers,
        loss_bundle=dirty.loss_bundle,
        device=dirty.device,
        use_amp=False,
        scaler=None,
        dtype="float16",
        grad_clip_generator=10.0,
        grad_clip_discriminator=10.0,
    )

    assert_module_parameters_equal(
        clean.models.generator,
        dirty.models.generator,
    )

    assert_module_parameters_equal(
        clean.models.discriminator,
        dirty.models.discriminator,
    )


def test_train_step_returns_structured_loss_metrics() -> None:
    fixture = make_training_fixture()

    result = vocoder_train_step(
        batch=fixture.batch,
        models=fixture.models,
        optimizers=fixture.optimizers,
        loss_bundle=fixture.loss_bundle,
        device=fixture.device,
        use_amp=False,
        scaler=None,
        dtype="float16",
        grad_clip_generator=10.0,
        grad_clip_discriminator=10.0,
    )

    expected_keys = {
        "loss_g_total",
        "loss_g_adversarial",
        "loss_g_feature_matching",
        "loss_g_mrstft",
        "loss_g_waveform",
        "loss_g_temporal_gradient",
        "loss_g_weighted/adversarial",
        "loss_g_weighted/feature_matching",
        "loss_g_weighted/mrstft",
        "loss_g_weighted/waveform",
        "loss_g_weighted/temporal_gradient",
        "loss_g_group/learned",
        "loss_g_group/time_micro",
        "loss_g_group/time_frequency",
        "loss_d_total",
        "loss_d_real",
        "loss_d_fake",
    }

    assert expected_keys.issubset(
        result.losses
    )
    
    
def test_tiny_discriminator_health_metadata_matches_outputs() -> None:
    fixture = make_training_fixture()
    discriminator = fixture.models.discriminator
    real_audio = fixture.batch["audio"]
    fake_audio = fixture.models.generator(fixture.batch["cqt"])

    outputs = discriminator(real_audio, fake_audio)

    assert len( discriminator.config.mpd.periods ) == len( outputs["mpd_real_outputs"] )
    assert len( discriminator.config.mpd.periods ) == len( outputs["mpd_fake_outputs"] )
    assert len( discriminator.config.mrd.resolutions ) == len( outputs["mrd_real_outputs"] )
    assert len( discriminator.config.mrd.resolutions ) == len( outputs["mrd_fake_outputs"] )


def test_micro_step_uses_configured_feature_matching_normalization() -> None:
    fixture = make_training_fixture(seed=1337)
    fixture.loss_bundle.config.feature_matching.normalization = "real_magnitude"
    fixture.loss_bundle.config.feature_matching.scale_floor = 0.01

    result = vocoder_train_micro_step(
        batch=fixture.batch,
        models=fixture.models,
        loss_bundle=fixture.loss_bundle,
        device=fixture.device,
        use_amp=False,
        scaler=None,
        dtype="float16",
        loss_divisor=1,
    )

    assert ( "loss_g_feature_matching" in result.losses )
    assert ( "loss_g_feature_matching_mpd" in result.losses )

    # Only MPD is enabled in the tiny fixture, so the combined
    # family-balanced loss must equal the MPD family loss.

    assert result.losses["loss_g_feature_matching"] == pytest.approx(
        result.losses[ "loss_g_feature_matching_mpd" ],
        rel=1e-6,
        abs=1e-7,
    )


def test_training_seed_uses_epoch() -> None:
    assert derive_training_seed(
        base_seed=1337,
        epoch=0,
    ) == 1337

    assert derive_training_seed(
        base_seed=1337,
        epoch=1,
    ) == 1338

    assert derive_training_seed(
        base_seed=1337,
        epoch=10,
    ) == 1347


def test_training_seed_uses_resume_step() -> None:
    seed = derive_training_seed(
        base_seed=1337,
        epoch=2,
        resume_global_step=10000,
    )

    assert seed == 11339


def test_resume_seed_changes_random_sequence() -> None:
    first_seed = derive_training_seed(
        base_seed=1337,
        epoch=2,
        resume_global_step=10000,
    )

    second_seed = derive_training_seed(
        base_seed=1337,
        epoch=2,
        resume_global_step=12000,
    )

    set_seed(first_seed)
    first_values = torch.rand(8)

    set_seed(second_seed)
    second_values = torch.rand(8)

    assert not torch.equal(
        first_values,
        second_values,
    )


def test_derived_seed_is_deterministic() -> None:
    seed = derive_training_seed(
        base_seed=1337,
        epoch=3,
        resume_global_step=5000,
    )

    set_seed(seed)
    first = torch.rand(8)

    set_seed(seed)
    second = torch.rand(8)

    assert torch.equal(first, second)


