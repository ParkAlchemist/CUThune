import pytest
import torch
from torch import nn

from shared.src.chiptune_shared.nn.blocks.convnext import ConvNeXtContextBlock1d, ConvNeXtContextTrunk1d, \
    compute_balanced_channel_splits, MultiKernelConvNeXtContextBlock1d, ConvNeXtContextBlock2d
from shared.src.chiptune_shared.nn.blocks.global_response_norm import GlobalResponseNorm1d
from vocoder.src.nes_vocoder.config.schema import VocoderGeneratorModelConfig

from vocoder.src.nes_vocoder.models.generator import CQTUHiFiGANGenerator

@pytest.mark.parametrize(
    ("kernel_size", "dilation"),
    [
        (3, 1),
        (7, 1),
        (7, 2),
        (15, 1),
    ],
)
def test_convnext_context_block_1d_preserves_shape_and_gradient(
    kernel_size: int,
    dilation: int,
) -> None:
    channels = 16

    block = ConvNeXtContextBlock1d(
        channels=channels,
        kernel_size=kernel_size,
        dilation=dilation,
        expansion_ratio=2,
    )

    x = torch.randn(
        2,
        channels,
        128,
        requires_grad=True,
    )

    y = block(x)
    y.square().mean().backward()

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    parameter_gradients = [
        parameter.grad
        for parameter in block.parameters()
        if parameter.requires_grad
    ]

    assert parameter_gradients
    assert all(
        gradient is not None for gradient in parameter_gradients
    )
    assert all(
        torch.isfinite(gradient).all() for gradient in parameter_gradients
    )


def test_convnext_context_trunk_1d_preserves_shape_and_gradient() -> None:
    trunk = ConvNeXtContextTrunk1d(
        channels=16,
        number_of_blocks=4,
        kernel_size=7,
        dilation=1,
        expansion_ratio=2,
    )

    x = torch.randn(
        2,
        16,
        128,
        requires_grad=True,
    )

    y = trunk(x)
    y.square().mean().backward()

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    assert len(trunk.blocks) == 4

    for block in trunk.blocks:
        for parameter in block.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def test_1d_block_rejects_even_kernel() -> None:
    with pytest.raises(
        ValueError,
        match="kernel_size must be odd, got 4",
    ):
        ConvNeXtContextBlock1d(
            channels=16,
            kernel_size=4,
        )


def test_context_trunk_1d_state_dict_roundtrip() -> None:
    first = ConvNeXtContextTrunk1d(
        channels=16,
        number_of_blocks=2,
        kernel_size=7,
        dilation=1,
    )

    second = ConvNeXtContextTrunk1d(
        channels=16,
        number_of_blocks=2,
        kernel_size=7,
        dilation=1,
    )

    second.load_state_dict(
        first.state_dict(),
        strict=True,
    )

    x = torch.randn(2, 16, 64)

    first.eval()
    second.eval()

    with torch.no_grad():
        first_output = first(x)
        second_output = second(x)

    assert torch.allclose(
        first_output,
        second_output,
        atol=1e-5,
        rtol=1e-4,
    )


def test_zero_layer_scale_makes_1d_block_identity() -> None:
    block = ConvNeXtContextBlock1d(
        channels=16,
        layer_scale_initial=0.0,
    )

    x = torch.randn(2, 16, 64)
    y = block(x)

    assert torch.equal(y, x)


def test_vocoder_has_context_trunk_1d() -> None:

    config = VocoderGeneratorModelConfig()

    config.context.enabled = True

    model = CQTUHiFiGANGenerator(
        cqt_bins=116,
        model_config=config
    )

    assert isinstance(model.context_trunk, ConvNeXtContextTrunk1d)


def test_vocoder_does_not_have_context_trunk_1d() -> None:

    config = VocoderGeneratorModelConfig()

    config.context.enabled = False

    model = CQTUHiFiGANGenerator(
        cqt_bins=116,
        model_config=config
    )

    assert not isinstance(model.context_trunk, ConvNeXtContextTrunk1d)


@pytest.mark.parametrize(
    ("channels", "branches", "expected"),
    [
        (16, 4, (4, 4, 4, 4)),
        (18, 4, (5, 5, 4, 4)),
        (10, 3, (4, 3, 3)),
        (7, 7, (1, 1, 1, 1, 1, 1, 1)),
    ],
)
def test_balanced_channel_splits(
    channels: int,
    branches: int,
    expected: tuple[int, ...],
) -> None:
    splits = compute_balanced_channel_splits(
        channels,
        branches,
    )

    assert splits == expected
    assert sum(splits) == channels
    assert all(split > 0 for split in splits)


def test_balanced_channel_splits_rejects_too_many_branches() -> None:
    with pytest.raises(
        ValueError,
        match="must be <= channels",
    ):
        compute_balanced_channel_splits(
            channels=3,
            number_of_branches=4,
        )


@pytest.mark.parametrize(
    "channels",
    [
        16,
        18,
        33,
    ],
)
def test_multi_kernel_block_1d_preserves_shape_and_gradients(
    channels: int,
) -> None:
    block = MultiKernelConvNeXtContextBlock1d(
        channels=channels,
        kernel_sizes=(3, 7, 15, 31),
        dilations=(1, 1, 1, 1),
        expansion_ratio=2,
    )

    x = torch.randn(
        2,
        channels,
        128,
        requires_grad=True,
    )

    y = block(x)
    loss = y.square().mean()
    loss.backward()

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, (
            f"Parameter {name!r} received no gradient."
        )
        assert torch.isfinite(parameter.grad).all()


def test_multi_kernel_1d_branches_are_depthwise() -> None:
    block = MultiKernelConvNeXtContextBlock1d(
        channels=18,
        kernel_sizes=(3, 7, 15, 31),
    )

    assert block.channel_splits == (5, 5, 4, 4)
    assert len(block.depthwise_branches) == 4

    for split, kernel_size, branch in zip(
        block.channel_splits,
        block.kernel_sizes,
        block.depthwise_branches,
    ):
        assert branch.in_channels == split
        assert branch.out_channels == split
        assert branch.groups == split
        assert branch.kernel_size == (kernel_size,)


def test_multi_kernel_block_1d_zero_layer_scale_is_identity() -> None:
    block = MultiKernelConvNeXtContextBlock1d(
        channels=16,
        kernel_sizes=(3, 7, 15, 31),
        layer_scale_initial=0.0,
    )

    x = torch.randn(2, 16, 128)
    y = block(x)

    assert torch.equal(y, x)


def test_multi_kernel_1d_branches_are_registered() -> None:
    block = MultiKernelConvNeXtContextBlock1d(
        channels=16,
        kernel_sizes=(3, 7, 15, 31),
    )

    assert isinstance(
        block.depthwise_branches,
        nn.ModuleList,
    )

    parameter_names = {
        name
        for name, _ in block.named_parameters()
    }

    for branch_index in range(4):
        assert any(
            name.startswith(
                f"depthwise_branches.{branch_index}."
            )
            for name in parameter_names
        )


def test_multi_kernel_block_1d_state_dict_roundtrip() -> None:
    first = MultiKernelConvNeXtContextBlock1d(
        channels=16,
        kernel_sizes=(3, 7, 15, 31),
    )

    second = MultiKernelConvNeXtContextBlock1d(
        channels=16,
        kernel_sizes=(3, 7, 15, 31),
    )

    second.load_state_dict(
        first.state_dict(),
        strict=True,
    )

    x = torch.randn(2, 16, 128)

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


def test_multi_kernel_trunk_1d_constructs_correct_blocks() -> None:
    trunk = ConvNeXtContextTrunk1d(
        channels=32,
        number_of_blocks=4,
        block_type="multi_kernel_convnext",
        multi_kernel_sizes=(3, 7, 15, 31),
        multi_kernel_dilations=(1, 1, 1, 1),
        expansion_ratio=2,
    )

    assert len(trunk.blocks) == 4

    assert all(
        isinstance(
            block,
            MultiKernelConvNeXtContextBlock1d,
        )
        for block in trunk.blocks
    )


def test_multi_kernel_trunk_1d_backward() -> None:
    trunk = ConvNeXtContextTrunk1d(
        channels=32,
        number_of_blocks=4,
        block_type="multi_kernel_convnext",
        multi_kernel_sizes=(3, 7, 15, 31),
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

    for parameter in trunk.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_generator_with_multi_kernel_context_1d(small_generator_config) -> None:
    config = small_generator_config

    config.context.enabled = True
    config.context.block_type = (
        "multi_kernel_convnext"
    )
    config.context.number_of_blocks = 2
    config.context.multi_kernel_sizes = (
        3,
        7,
        15,
        31,
    )
    config.context.multi_kernel_dilations = (
        1,
        1,
        1,
        1,
    )

    generator = CQTUHiFiGANGenerator(
        cqt_bins=116,
        model_config=config,
    )

    assert all(isinstance(block, MultiKernelConvNeXtContextBlock1d) for block in generator.context_trunk.blocks)

    cqt = torch.randn(
        1,
        116,
        8,
        requires_grad=True,
    )

    waveform = generator(cqt)

    expected_length = (
        8 * generator.total_upsample_factor
    )

    assert waveform.shape == (
        1,
        1,
        expected_length,
    )
    assert torch.isfinite(waveform).all()

    waveform.square().mean().backward()

    assert cqt.grad is not None
    assert torch.isfinite(cqt.grad).all()

@pytest.fixture
def small_generator_config():
    return VocoderGeneratorModelConfig()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_multi_kernel_context_1d_generator_runs_on_cuda(small_generator_config) -> None:
    device = torch.device("cuda")

    config = small_generator_config

    config.context.enabled = True
    config.context.block_type = (
        "multi_kernel_convnext"
    )
    config.context.number_of_blocks = 2
    config.context.multi_kernel_sizes = (
        3,
        7,
        15,
        31,
    )

    generator = CQTUHiFiGANGenerator(
        cqt_bins=116,
        model_config=config,
    ).to(device)

    cqt = torch.randn(
        1,
        116,
        8,
        device=device,
        requires_grad=True,
    )

    waveform = generator(cqt)
    waveform.mean().backward()

    assert waveform.device.type == device.type
    assert cqt.grad is not None

    for parameter in generator.parameters():
        assert parameter.device.type == device.type


def test_convnext_block_1d_constructs_grn_1d_in_expanded_space() -> None:
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


def test_convnext_block_1d_disables_grn_1d_with_identity() -> None:
    block = ConvNeXtContextBlock1d(
        channels=16,
        expansion_ratio=2,
        use_grn=False,
    )

    assert isinstance(
        block.grn,
        nn.Identity,
    )

def test_multi_kernel_block_1d_constructs_grn_1d() -> None:
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
def test_convnext_blocks_1d_with_grn_1d_backward(
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


def test_convnext_2d_block_preserves_shape() -> None:
    block = ConvNeXtContextBlock2d(
        channels=32,
        use_grn=True,
    )
    x = torch.randn(2, 32, 29, 172)

    y = block(x)

    assert y.shape == x.shape


def test_convnext_2d_initial_output_is_close_to_input() -> None:
    block = ConvNeXtContextBlock2d(
        channels=32,
        use_grn=True,
        layer_scale_initial=1e-6,
    )
    x = torch.randn(2, 32, 29, 172)

    y = block(x)

    torch.testing.assert_close(
        y,
        x,
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "use_grn",
    [False, True],
)
def test_convnext_2d_backward_is_finite(
    use_grn: bool,
) -> None:
    block = ConvNeXtContextBlock2d(
        channels=32,
        use_grn=use_grn,
    )
    x = torch.randn(
        2,
        32,
        29,
        172,
        requires_grad=True,
    )

    loss = block(x).square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_convnext_2d_rejects_wrong_channel_count() -> None:
    block = ConvNeXtContextBlock2d(
        channels=32,
    )
    x = torch.randn(2, 16, 29, 172)

    with pytest.raises(
        RuntimeError,
        match="channel mismatch",
    ):
        block(x)


def test_convnext_2d_rejects_non_4d_input() -> None:
    block = ConvNeXtContextBlock2d(
        channels=32,
    )
    x = torch.randn(2, 32, 172)

    with pytest.raises(
        RuntimeError,
        match="4 dimensional",
    ):
        block(x)

