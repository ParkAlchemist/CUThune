import pytest
import torch

from vocoder.src.nes_vocoder.models.alias_free import (
    AliasFreeActivation1d,
    DownSample1d,
    UpSample1d,
    design_lowpass_filter,
)


def test_lowpass_filter_is_finite_and_normalized():
    kernel = design_lowpass_filter(
        cutoff=0.2,
        transition_bandwidth=0.05,
        kernel_size=12,
    )

    assert kernel.shape == (12,)
    assert torch.isfinite(kernel).all()
    assert kernel.sum().item() == pytest.approx(
        1.0,
        rel=1e-5,
        abs=1e-6,
    )


@pytest.mark.parametrize("kernel_size", [11, 12, 13])
def test_lowpass_filter_is_symmetric(
    kernel_size: int,
):
    kernel = design_lowpass_filter(
        cutoff=0.2,
        transition_bandwidth=0.05,
        kernel_size=kernel_size,
    )

    assert torch.allclose(
        kernel,
        kernel.flip(0),
        atol=1e-6,
        rtol=1e-5,
    )


def test_invalid_lowpass_parameters_raise():
    with pytest.raises(ValueError):
        design_lowpass_filter(
            cutoff=0.0,
            transition_bandwidth=0.05,
            kernel_size=12,
        )

    with pytest.raises(ValueError):
        design_lowpass_filter(
            cutoff=0.3,
            transition_bandwidth=0.25,
            kernel_size=12,
        )


@pytest.mark.parametrize("length", [31, 32, 63, 64, 129])
def test_upsample_doubles_length(length: int):
    module = UpSample1d(
        ratio=2,
        kernel_size=12,
    )

    x = torch.randn(2, 8, length)
    y = module(x)

    assert y.shape == (2, 8, length * 2)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("length", [32, 64, 128, 258])
def test_downsample_halves_even_length(length: int):
    module = DownSample1d(
        ratio=2,
        kernel_size=12,
    )

    x = torch.randn(2, 8, length)
    y = module(x)

    assert y.shape == (2, 8, length // 2)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("length", [32, 63, 128])
def test_alias_free_wrapper_preserves_shape(length: int):
    module = AliasFreeActivation1d(
        torch.nn.Identity(),
        upsample_ratio=2,
        downsample_ratio=2,
        upsample_kernel_size=12,
        downsample_kernel_size=12,
    )

    x = torch.randn(2, 8, length)
    y = module(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


from vocoder.src.nes_vocoder.models.generator import SnakeBeta


def test_alias_free_snake_beta_shape_and_gradient():
    module = AliasFreeActivation1d(
        SnakeBeta(
            channels=8,
            alpha_logscale=True,
        ),
        upsample_ratio=2,
        downsample_ratio=2,
        upsample_kernel_size=12,
        downsample_kernel_size=12,
    )

    x = torch.randn(
        2,
        8,
        64,
        requires_grad=True,
    )

    y = module(x)
    loss = y.square().mean()
    loss.backward()

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    parameter_grads = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad
    ]

    assert parameter_grads
    assert all(gradient is not None for gradient in parameter_grads)
    assert all(torch.isfinite(gradient).all() for gradient in parameter_grads)


def test_alias_free_activation_state_dict_roundtrip():
    first = AliasFreeActivation1d(
        SnakeBeta(
            channels=4,
            alpha_logscale=True,
        ),
        upsample_ratio=2,
        downsample_ratio=2,
        upsample_kernel_size=12,
        downsample_kernel_size=12,
    )

    state = first.state_dict()

    second = AliasFreeActivation1d(
        SnakeBeta(
            channels=4,
            alpha_logscale=True,
        ),
        upsample_ratio=2,
        downsample_ratio=2,
        upsample_kernel_size=12,
        downsample_kernel_size=12,
    )

    second.load_state_dict(state, strict=True)

    x = torch.randn(1, 4, 64)

    first.eval()
    second.eval()

    with torch.no_grad():
        first_output = first(x)
        second_output = second(x)

    assert torch.allclose(
        first_output,
        second_output,
        atol=1e-6,
        rtol=1e-5,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_alias_free_snake_beta_cuda():
    module = AliasFreeActivation1d(
        SnakeBeta(
            channels=8,
            alpha_logscale=True,
        ),
        upsample_ratio=2,
        downsample_ratio=2,
        upsample_kernel_size=12,
        downsample_kernel_size=12,
    ).cuda()

    x = torch.randn(
        2,
        8,
        128,
        device="cuda",
        requires_grad=True,
    )

    y = module(x)
    y.mean().backward()

    assert y.is_cuda
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert x.grad is not None


def test_snake_beta_logscale_initialization():
    module = SnakeBeta(
        channels=4,
        alpha=2.0,
        beta=0.5,
        alpha_logscale=True,
    )

    alpha, beta = module.effective_parameters()

    assert torch.allclose(
        alpha,
        torch.full_like(alpha, 2.0),
    )

    assert torch.allclose(
        beta,
        torch.full_like(beta, 0.5),
    )


def test_snake_beta_non_logscale_matches_original_formula():
    module = SnakeBeta(
        channels=2,
        alpha=1.5,
        beta=0.75,
        alpha_logscale=False,
    )

    x = torch.randn(2, 2, 32)

    expected = (
        x
        + torch.sin(1.5 * x).square()
        / 0.75
    )

    actual = module(x)

    assert torch.allclose(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-5,
    )


def test_snake_beta_log_and_linear_initial_outputs_match():
    x = torch.randn(2, 4, 64)

    log_module = SnakeBeta(
        channels=4,
        alpha=1.0,
        beta=1.0,
        alpha_logscale=True,
    )

    linear_module = SnakeBeta(
        channels=4,
        alpha=1.0,
        beta=1.0,
        alpha_logscale=False,
    )

    log_output = log_module(x)
    linear_output = linear_module(x)

    assert torch.allclose(
        log_output,
        linear_output,
        atol=1e-6,
        rtol=1e-5,
    )


def test_snake_beta_logscale_gradients():
    module = SnakeBeta(
        channels=8,
        alpha=1.0,
        beta=1.0,
        alpha_logscale=True,
    )

    x = torch.randn(
        2,
        8,
        64,
        requires_grad=True,
    )

    y = module(x)
    y.square().mean().backward()

    assert x.grad is not None
    assert module.alpha.grad is not None
    assert module.beta.grad is not None

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(module.alpha.grad).all()
    assert torch.isfinite(module.beta.grad).all()

