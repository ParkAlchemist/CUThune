import torch

from cut.src.nes_cut.models.feature_sampler import FeatureSampler


def test_sampler_reuses_patch_ids() -> None:
    sampler = FeatureSampler(
        {
            "down_1": 128,
            "down_2": 256,
        },
        projection_dim=64,
    )

    source = {
        "down_1": torch.randn(
            2,
            128,
            8,
            12,
        ),
        "down_2": torch.randn(
            2,
            256,
            4,
            6,
        ),
    }

    generated = {
        name: tensor.clone()
        for name, tensor in source.items()
    }

    source_projected, patch_ids = sampler(
        source,
        num_patches=16,
    )

    generated_projected, reused_ids = sampler(
        generated,
        num_patches=16,
        patch_ids=patch_ids,
    )

    for name in patch_ids:
        torch.testing.assert_close(
            reused_ids[name],
            patch_ids[name],
        )
        torch.testing.assert_close(
            generated_projected[name],
            source_projected[name],
        )


def test_sampler_output_shape() -> None:
    sampler = FeatureSampler(
        {"layer": 32},
        projection_dim=64,
    )

    features = {
        "layer": torch.randn(
            3,
            32,
            8,
            10,
        )
    }

    projected, ids = sampler(
        features,
        num_patches=20,
    )

    assert projected["layer"].shape == (
        3,
        20,
        64,
    )
    assert ids["layer"].shape == (
        3,
        20,
    )


def test_sampler_clamps_patch_count_to_spatial_size() -> None:
    sampler = FeatureSampler(
        {"layer": 16},
        projection_dim=32,
    )

    features = {
        "layer": torch.randn(
            2,
            16,
            3,
            4,
        )
    }

    projected, ids = sampler(
        features,
        num_patches=100,
    )

    assert projected["layer"].shape == (
        2,
        12,
        32,
    )
    assert ids["layer"].shape == (
        2,
        12,
    )


def test_projected_features_have_unit_norm() -> None:
    sampler = FeatureSampler(
        {"layer": 16},
        projection_dim=32,
    )

    projected, _ = sampler(
        {
            "layer": torch.randn(
                2,
                16,
                4,
                5,
            )
        },
        num_patches=10,
    )

    norms = torch.linalg.vector_norm(
        projected["layer"],
        dim=-1,
    )

    torch.testing.assert_close(
        norms,
        torch.ones_like(norms),
        rtol=1e-5,
        atol=1e-6,
    )


def test_projection_heads_receive_gradients() -> None:
    sampler = FeatureSampler(
        {"layer": 16},
        projection_dim=32,
    )

    feature = torch.randn(
        2,
        16,
        4,
        5,
        requires_grad=True,
    )

    projected, _ = sampler(
        {"layer": feature},
        num_patches=10,
    )

    loss = projected["layer"].square().mean()
    loss.backward()

    assert feature.grad is not None
    assert torch.isfinite(feature.grad).all()

    for parameter in sampler.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_sampler_supports_batch_size_one() -> None:
    sampler = FeatureSampler(
        {"layer": 16},
        projection_dim=32,
    )

    projected, ids = sampler(
        {
            "layer": torch.randn(
                1,
                16,
                4,
                5,
            )
        },
        num_patches=10,
    )

    assert projected["layer"].shape == (
        1,
        10,
        32,
    )
    assert ids["layer"].shape == (
        1,
        10,
    )


