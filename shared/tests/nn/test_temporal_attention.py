import pytest
import torch
from torch import nn

from shared.src.chiptune_shared.nn.blocks.convnext import ConvNeXtContextTrunk1d
from shared.src.chiptune_shared.nn.blocks.temporal_attention import RotaryPositionEmbedding, RoPETemporalSelfAttention1d
from vocoder.src.nes_vocoder.config.schema import ContextAttentionConfig


def test_rope_preserves_shape_and_norm() -> None:
    rope = RotaryPositionEmbedding(
        head_dimension=16,
    )

    x = torch.randn(
        2,
        4,
        32,
        16,
    )

    y = rope(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    input_norm = x.square().sum(dim=-1)
    output_norm = y.square().sum(dim=-1)

    assert torch.allclose(
        input_norm,
        output_norm,
        atol=1e-5,
        rtol=1e-5,
    )


def test_rope_position_zero_is_identity() -> None:
    rope = RotaryPositionEmbedding(
        head_dimension=16,
    )

    x = torch.randn(
        2,
        4,
        32,
        16,
    )

    y = rope(x)

    assert torch.allclose(
        y[:, :, 0, :],
        x[:, :, 0, :],
        atol=1e-6,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    ("channels", "heads"),
    [
        (32, 4),
        (64, 8),
        (256, 8),
    ],
)
def test_rope_attention_preserves_shape_and_gradient(
    channels: int,
    heads: int,
) -> None:
    attention = RoPETemporalSelfAttention1d(
        channels=channels,
        number_of_heads=heads,
        dropout=0.0,
    )

    x = torch.randn(
        2,
        channels,
        64,
        requires_grad=True,
    )

    y = attention(x)
    y.square().mean().backward()

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, parameter in attention.named_parameters():
        assert parameter.grad is not None, (
            f"Parameter {name!r} received no gradient."
        )
        assert torch.isfinite(parameter.grad).all()


def test_attention_zero_layer_scale_is_identity() -> None:
    attention = RoPETemporalSelfAttention1d(
        channels=32,
        number_of_heads=4,
        layer_scale_initial=0.0,
    )

    x = torch.randn(2, 32, 64)
    y = attention(x)

    assert torch.equal(y, x)


@pytest.mark.parametrize(
    "sequence_length",
    [
        1,
        7,
        32,
        128,
        192,
    ],
)
def test_attention_supports_variable_lengths(
    sequence_length: int,
) -> None:
    attention = RoPETemporalSelfAttention1d(
        channels=32,
        number_of_heads=4,
    )

    x = torch.randn(
        2,
        32,
        sequence_length,
    )

    y = attention(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_attention_rejects_non_divisible_head_count() -> None:
    with pytest.raises(
        ValueError,
        match="divisible",
    ):
        RoPETemporalSelfAttention1d(
            channels=30,
            number_of_heads=8,
        )


def test_attention_rejects_odd_head_dimension() -> None:
    with pytest.raises(
        ValueError,
        match="even",
    ):
        RoPETemporalSelfAttention1d(
            channels=24,
            number_of_heads=8,
        )


def test_context_trunk_registers_attention() -> None:
    trunk = ConvNeXtContextTrunk1d(
        channels=32,
        number_of_blocks=4,
        block_type="convnext",
        multi_kernel_sizes=(7, 7, 7, 7),
        multi_kernel_dilations=(1, 1, 1, 1),
        attn_config=ContextAttentionConfig(
            enabled=True,
            placement=2,
            number_of_heads=4,
        ),
    )

    assert isinstance(
        trunk.attention,
        RoPETemporalSelfAttention1d,
    )

    child_names = dict(
        trunk.named_children()
    )

    assert "attention" in child_names


class CountingIdentity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        self.calls += 1
        return x


@pytest.mark.parametrize(
    "placement",
    [
        0,
        1,
        2,
        4,
    ],
)
def test_context_attention_is_called_once(
    placement: int,
) -> None:
    trunk = ConvNeXtContextTrunk1d(
        channels=16,
        number_of_blocks=4,
        block_type="convnext",
        multi_kernel_sizes=(7, 7, 7, 7),
        multi_kernel_dilations=(1, 1, 1, 1),
        attn_config=ContextAttentionConfig(
            enabled=True,
            placement=placement,
            number_of_heads=4,
        ),
    )

    counter = CountingIdentity()
    trunk.attention = counter

    x = torch.randn(2, 16, 64)
    trunk(x)

    assert counter.calls == 1


def test_context_trunk_with_attention_backward() -> None:
    trunk = ConvNeXtContextTrunk1d(
        channels=32,
        number_of_blocks=4,
        block_type="multi_kernel_convnext",
        multi_kernel_sizes=(3, 7, 15, 31),
        multi_kernel_dilations=(1, 1, 1, 1),
        attn_config=ContextAttentionConfig(
            enabled=True,
            placement=2,
            number_of_heads=4,
        ),
    )

    x = torch.randn(
        2,
        32,
        128,
        requires_grad=True,
    )

    y = trunk(x)
    y.square().mean().backward()

    assert y.shape == x.shape

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, parameter in trunk.named_parameters():
        assert parameter.grad is not None, (
            f"Parameter {name!r} received no gradient."
        )
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_context_attention_runs_on_cuda() -> None:
    device = torch.device("cuda")

    attention = RoPETemporalSelfAttention1d(
        channels=64,
        number_of_heads=8,
    ).to(device)

    x = torch.randn(
        2,
        64,
        128,
        device=device,
        requires_grad=True,
    )

    y = attention(x)
    y.square().mean().backward()

    assert y.device.type == device.type
    assert x.grad is not None
    assert x.grad.device.type == device.type

    for parameter in attention.parameters():
        assert parameter.device.type == device.type

    for buffer in attention.buffers():
        assert buffer.device.type == device.type


