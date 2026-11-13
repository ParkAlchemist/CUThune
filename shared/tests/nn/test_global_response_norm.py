import pytest
import torch

from shared.src.chiptune_shared.nn.blocks.global_response_norm import GlobalResponseNorm1d, GlobalResponseNorm2d


def test_grn_1d_preserves_shape() -> None:
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


def test_grn_1d_is_identity_at_initialization() -> None:
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


def test_grn_1d_gradients_are_finite() -> None:
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


def test_grn_1d_applies_channel_response_modulation() -> None:
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


def test_grn_1d_handles_zero_input() -> None:
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


def test_grn_1d_rejects_wrong_rank() -> None:
    grn = GlobalResponseNorm1d(
        channels=32,
    )

    x = torch.randn(2, 32)

    with pytest.raises(
        ValueError,
        match=r"3 dimensions",
    ):
        grn(x)


def test_grn_1d_rejects_channel_mismatch() -> None:
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


def test_grn_2d_initially_behaves_as_identity() -> None:
    grn = GlobalResponseNorm2d(channels=16)
    x = torch.randn(2, 8, 12, 16)

    y = grn(x)

    torch.testing.assert_close(y, x)


def test_grn_2d_preserves_shape() -> None:
    grn = GlobalResponseNorm2d(channels=16)
    x = torch.randn(2, 8, 12, 16)

    assert grn(x).shape == x.shape


def test_grn_2d_backward_is_finite() -> None:
    grn = GlobalResponseNorm2d(channels=16)
    x = torch.randn(
        2,
        8,
        12,
        16,
        requires_grad=True,
    )

    loss = grn(x).square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert grn.gamma.grad is not None
    assert grn.beta.grad is not None
    assert torch.isfinite(grn.gamma.grad).all()
    assert torch.isfinite(grn.beta.grad).all()


def test_grn_2d_handles_zero_input() -> None:
    grn = GlobalResponseNorm2d(channels=16)
    x = torch.zeros(2, 8, 12, 16)

    y = grn(x)

    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, x)

