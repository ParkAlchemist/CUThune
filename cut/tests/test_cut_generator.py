import pytest
import torch

from cut.src.nes_cut.models.generator import CUTGenerator, CUTGeneratorConfig


def test_generator_training_geometry() -> None:
    model = CUTGenerator(
        CUTGeneratorConfig()
    )

    x = torch.randn(
        2,
        1,
        116,
        688,
    )

    output, features = model.forward_with_features(
        x,
        feature_layers=(
            "stem",
            "down_1",
            "down_2",
            "bottleneck_3",
        ),
    )

    assert output.shape == x.shape

    assert features["stem"].shape == (
        2,
        64,
        116,
        688,
    )

    assert features["down_1"].shape == (
        2,
        128,
        58,
        344,
    )

    assert features["down_2"].shape == (
        2,
        256,
        29,
        172,
    )

    assert features["bottleneck_3"].shape == (
        2,
        256,
        29,
        172,
    )


@pytest.mark.parametrize(
    "frames",
    [687, 688, 689, 690, 691],
)
def test_generator_preserves_arbitrary_time_length(
    frames: int,
) -> None:
    model = CUTGenerator(
        CUTGeneratorConfig()
    )

    x = torch.randn(
        1,
        1,
        116,
        frames,
    )

    y = model(x)

    assert y.shape == x.shape


def test_generator_preserves_requested_feature_order() -> None:
    model = CUTGenerator(
        CUTGeneratorConfig()
    )

    x = torch.randn(1, 1, 116, 688)

    requested = (
        "bottleneck_2",
        "stem",
        "down_1",
    )

    _, features = model.forward_with_features(
        x,
        feature_layers=requested,
    )

    assert tuple(features) == requested


def test_generator_rejects_unknown_feature() -> None:
    model = CUTGenerator(
        CUTGeneratorConfig()
    )

    x = torch.randn(1, 1, 116, 688)

    with pytest.raises(
        ValueError,
        match="Unknown feature layers",
    ):
        model.forward_with_features(
            x,
            feature_layers=("missing",),
        )


def test_generator_rejects_duplicate_feature_names() -> None:
    model = CUTGenerator(
        CUTGeneratorConfig()
    )

    x = torch.randn(1, 1, 116, 688)

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        model.forward_with_features(
            x,
            feature_layers=(
                "down_1",
                "down_1",
            ),
        )


def test_generator_backward_is_finite() -> None:
    model = CUTGenerator(
        CUTGeneratorConfig()
    )

    x = torch.randn(
        1,
        1,
        116,
        688,
        requires_grad=True,
    )

    y = model(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    named_parameters = dict(
        model.named_parameters()
    )

    required = (
        "stem.conv.weight",
        "downs.0.projection.conv.weight",
        "downs.1.projection.conv.weight",
        "ups.0.conv.conv.weight",
        "ups.1.conv.conv.weight",
        "head.conv.weight",
    )

    for name in required:
        gradient = named_parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name


def test_output_head_does_not_collapse() -> None:
    model = CUTGenerator(CUTGeneratorConfig())
    x = torch.randn(2, 1, 116, 688)

    y = model(x)

    assert y.std().item() > 1e-6


def test_feature_names_follow_configuration() -> None:
    model = CUTGenerator(
        CUTGeneratorConfig(
            num_sampling_blocks=2,
            num_bottleneck_blocks=5,
        )
    )

    assert model.available_feature_layers == (
        "stem",
        "down_1",
        "down_2",
        "bottleneck_1",
        "bottleneck_2",
        "bottleneck_3",
        "bottleneck_4",
        "bottleneck_5",
    )


