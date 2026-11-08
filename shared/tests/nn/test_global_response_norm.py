import pytest
import torch
from torch import nn

from shared.src.chiptune_shared.nn.blocks.convnext import GlobalResponseNorm1d, ConvNeXtContextBlock1d, \
    MultiKernelConvNeXtContextBlock1d


def test_grn_preserves_shape() -> None:
    grn = GlobalResponseNorm1d(
        channels=32,
    )

    x = torch.randn(
        2,
        128,
        32,
    )

    y = grn(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_grn_is_identity_at_initialization() -> None:
    grn = GlobalResponseNorm1d(
        channels=32,
    )

    x = torch.randn(
        2,
        128,
        32,
    )

    y = grn(x)

    assert torch.equal(y, x)


def test_grn_gradients_are_finite() -> None:
    grn = GlobalResponseNorm1d(
        channels=32,
    )

    x = torch.randn(
        2,
        128,
        32,
        requires_grad=True,
    )

    y = grn(x)
    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert grn.gamma.grad is not None
    assert torch.isfinite(grn.gamma.grad).all()

    assert grn.beta.grad is not None
    assert torch.isfinite(grn.beta.grad).all()


def test_grn_applies_channel_response_modulation() -> None:
    grn = GlobalResponseNorm1d(
        channels=2,
        eps=1e-6,
    )

    with torch.no_grad():
        grn.gamma.fill_(1.0)
        grn.beta.zero_()

    x = torch.tensor(
        [
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
            ]
        ]
    )

    y = grn(x)

    global_response = torch.linalg.vector_norm(
        x,
        ord=2,
        dim=1,
        keepdim=True,
    )

    normalized_response = (
        global_response
        / global_response.mean(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
    )

    expected = (
        x
        + x * normalized_response
    )

    assert torch.allclose(
        y,
        expected,
        atol=1e-6,
        rtol=1e-5,
    )


def test_grn_handles_zero_input() -> None:
    grn = GlobalResponseNorm1d(
        channels=32,
    )

    x = torch.zeros(
        2,
        128,
        32,
    )

    y = grn(x)

    assert torch.isfinite(y).all()
    assert torch.equal(y, x)


def test_grn_rejects_wrong_rank() -> None:
    grn = GlobalResponseNorm1d(
        channels=32,
    )

    x = torch.randn(2, 32)

    with pytest.raises(
        ValueError,
        match=r"3 dimensions",
    ):
        grn(x)


def test_grn_rejects_channel_mismatch() -> None:
    grn = GlobalResponseNorm1d(
        channels=32,
    )

    x = torch.randn(
        2,
        128,
        16,
    )

    with pytest.raises(
        RuntimeError,
        match="channel mismatch",
    ):
        grn(x)


def test_convnext_block_constructs_grn_in_expanded_space() -> None:
    block = ConvNeXtContextBlock1d(
        channels=16,
        expansion_ratio=2,
        use_grn=True,
    )

    assert isinstance(
        block.grn,
        GlobalResponseNorm1d,
    )

    assert block.grn.channels == 32


def test_convnext_block_disables_grn_with_identity() -> None:
    block = ConvNeXtContextBlock1d(
        channels=16,
        expansion_ratio=2,
        use_grn=False,
    )

    assert isinstance(
        block.grn,
        nn.Identity,
    )

def test_multi_kernel_block_constructs_grn() -> None:
    block = MultiKernelConvNeXtContextBlock1d(
        channels=16,
        kernel_sizes=(3, 7, 15, 31),
        expansion_ratio=2,
        use_grn=True,
    )

    assert isinstance(
        block.grn,
        GlobalResponseNorm1d,
    )

    assert block.grn.channels == 32


@pytest.mark.parametrize(
    "block_class",
    [
        ConvNeXtContextBlock1d,
        MultiKernelConvNeXtContextBlock1d,
    ],
)
def test_convnext_blocks_with_grn_backward(
    block_class,
) -> None:
    kwargs = {
        "channels": 16,
        "expansion_ratio": 2,
        "use_grn": True,
    }

    if block_class is ConvNeXtContextBlock1d:
        kwargs["kernel_size"] = 7
    else:
        kwargs["kernel_sizes"] = (3, 7, 15, 31)

    block = block_class(**kwargs)

    x = torch.randn(
        2,
        16,
        128,
        requires_grad=True,
    )

    y = block(x)
    y.square().mean().backward()

    assert y.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, (
            f"Parameter {name!r} received no gradient."
        )
        assert torch.isfinite(parameter.grad).all()


