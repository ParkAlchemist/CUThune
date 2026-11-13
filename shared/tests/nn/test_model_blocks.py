import torch
from torch import nn

from cut.src.nes_cut.models.blocks.patch_discriminator import PatchDiscriminatorBlock
from shared.src.chiptune_shared.nn.blocks.channel_attention import SqueezeExciteBlock
from shared.src.chiptune_shared.nn.blocks.conv_norm_act import ConvNormAct
from shared.src.chiptune_shared.nn.blocks.downsample import DownsampleBlock, PixelUnshuffle
from shared.src.chiptune_shared.nn.blocks.residual import ResidualBlock
from shared.src.chiptune_shared.nn.blocks.spatial_attention import SpatialSelfAttention
from shared.src.chiptune_shared.nn.blocks.upsample import UpsampleBlock
from shared.src.chiptune_shared.nn.model_utils import CQTBufferPad, get_padding_layer


def test_cqt_buffer_pad_shape():
    x = torch.randn(2, 8, 16, 32)

    pad = CQTBufferPad(
        pad_time_left=1,
        pad_time_right=2,
        pad_freq_low=3,
        pad_freq_high=4,
    )

    y = pad(x)

    assert y.shape == (2, 8, 16 + 3 + 4, 32 + 1 + 2)


def test_get_padding_layer_cqt_int():
    x = torch.randn(2, 8, 16, 32)

    pad = get_padding_layer(1, padding_mode="cqt")
    y = pad(x)

    assert y.shape == (2, 8, 18, 34)


def test_get_padding_layer_cqt_tuple():
    x = torch.randn(2, 8, 16, 32)

    pad = get_padding_layer((1, 2, 3, 4), padding_mode="cqt")
    y = pad(x)

    assert y.shape == (2, 8, 23, 35)


def test_squeeze_excite_shape_and_gradient():
    block = SqueezeExciteBlock(channels=8, reduction=16)

    x = torch.randn(2, 8, 16, 32, requires_grad=True)
    y = block(x)

    assert y.shape == x.shape

    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_squeeze_excite_handles_small_channel_count():
    block = SqueezeExciteBlock(channels=4, reduction=16)

    x = torch.randn(2, 4, 8, 8)
    y = block(x)

    assert y.shape == x.shape


def test_base_blocks_shape_flow():
    x = torch.randn(2, 1, 96, 172)

    stem = ConvNormAct(
        in_channels=1,
        out_channels=32,
        kernel_size=7,
        padding=3,
        norm="instance",
        activation="relu",
        padding_mode="reflect",
    )

    down1 = DownsampleBlock(32, 64)
    down2 = DownsampleBlock(64, 128)

    res = ResidualBlock(
        channels=128,
        dropout=0.1,
        norm="instance",
        padding_mode="reflect",
    )

    attn = SpatialSelfAttention(128)

    up1 = UpsampleBlock(128, 64)
    up2 = UpsampleBlock(64, 32)

    with torch.no_grad():
        h0 = stem(x)
        h1 = down1(h0)
        h2 = down2(h1)
        h3 = res(h2)
        h4 = attn(h3)
        h5 = up1(h4)
        h6 = up2(h5)

    assert h0.shape == (2, 32, 96, 172)
    assert h1.shape == (2, 64, 48, 86)
    assert h2.shape == (2, 128, 24, 43)
    assert h3.shape == h2.shape
    assert h4.shape == h2.shape
    assert h5.shape == (2, 64, 48, 86)
    assert h6.shape == (2, 32, 96, 172)


def test_patch_discriminator_block_shape():
    x = torch.randn(2, 1, 96, 172)

    block = PatchDiscriminatorBlock(
        in_channels=1,
        out_channels=32,
        use_norm=False,
    )

    with torch.no_grad():
        y = block(x)

    assert y.ndim == 4
    assert y.shape[0] == 2
    assert y.shape[1] == 32


def test_custom_pixel_unshuffle_matches_pytorch():
    custom = PixelUnshuffle(ratio=2)
    reference = nn.PixelUnshuffle(2)

    x = torch.randn(2, 3, 8, 12)

    torch.testing.assert_close(
        custom(x),
        reference(x),
    )