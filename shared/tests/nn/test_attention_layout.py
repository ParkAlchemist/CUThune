import pytest

from src.models.cyclegan import (
    GeneratorConfig,
    split_resblocks_for_balanced_attention,
    build_bottleneck_layers,
)
from src.models.blocks import ResidualBlock
from src.models.blocks import SpatialSelfAttention


def layer_names(layers):
    names = []
    for layer in layers:
        if isinstance(layer, ResidualBlock):
            names.append("res")
        elif isinstance(layer, SpatialSelfAttention):
            names.append("att")
        else:
            names.append(type(layer).__name__)
    return names


def test_balanced_split_even():
    assert split_resblocks_for_balanced_attention(6, 2) == [2, 2, 2]


def test_balanced_split_uneven():
    assert split_resblocks_for_balanced_attention(7, 2) == [3, 2, 2]


def test_balanced_attention_layout_6_res_2_att():
    cfg = GeneratorConfig(
        num_res_blocks=6,
        use_attention=True,
        num_att_blocks=2,
        attention_position="balanced",
    )

    layers = build_bottleneck_layers(channels=128, config=cfg)

    assert layer_names(layers) == [
        "res", "res",
        "att",
        "res", "res",
        "att",
        "res", "res",
    ]


def test_middle_attention_layout_multiple_attention_blocks():
    cfg = GeneratorConfig(
        num_res_blocks=6,
        use_attention=True,
        num_att_blocks=2,
        attention_position="middle",
    )

    layers = build_bottleneck_layers(channels=128, config=cfg)

    assert layer_names(layers) == [
        "res", "res", "res",
        "att", "att",
        "res", "res", "res",
    ]


def test_invalid_attention_count_raises():
    cfg = GeneratorConfig(
        num_res_blocks=2,
        use_attention=True,
        num_att_blocks=3,
        attention_position="balanced",
    )

    with pytest.raises(ValueError):
        build_bottleneck_layers(channels=128, config=cfg)

