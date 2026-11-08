from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from cut.src.nes_cut.models.blocks.patch_discriminator import PatchDiscriminatorBlock
from shared.src.chiptune_shared.nn.blocks.conv_norm_act import ConvNormAct
from shared.src.chiptune_shared.nn.blocks.downsample import DownsampleBlock
from shared.src.chiptune_shared.nn.blocks.residual import ResidualBlock
from shared.src.chiptune_shared.nn.blocks.spatial_attention import SpatialSelfAttention
from shared.src.chiptune_shared.nn.blocks.upsample import UpsampleBlock

import torch
import torch.nn as nn

from shared.src.chiptune_shared.nn.model_utils import (
    count_parameters,
    init_weights,
)


def main() -> None:
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

    disc_block = PatchDiscriminatorBlock(
        in_channels=1,
        out_channels=32,
        use_norm=False,
    )

    model = nn.Sequential(stem, down1, down2, res, attn, up1, up2)
    model.apply(lambda m: init_weights(m, init_type="normal", gain=0.02))

    with torch.no_grad():
        h0 = stem(x)
        h1 = down1(h0)
        h2 = down2(h1)
        h3 = res(h2)
        h4 = attn(h3)
        h5 = up1(h4)
        h6 = up2(h5)
        d = disc_block(x)

    print("Input:", tuple(x.shape))
    print("Stem:", tuple(h0.shape))
    print("Down1:", tuple(h1.shape))
    print("Down2:", tuple(h2.shape))
    print("Res:", tuple(h3.shape))
    print("Attention:", tuple(h4.shape))
    print("Up1:", tuple(h5.shape))
    print("Up2:", tuple(h6.shape))
    print("Disc block:", tuple(d.shape))
    print("Params:", count_parameters(model))

    assert h0.shape == (2, 32, 96, 172)
    assert h1.shape == (2, 64, 48, 86)
    assert h2.shape == (2, 128, 24, 43)
    assert h3.shape == h2.shape
    assert h4.shape == h2.shape

    # Upsampling from odd width 43 produces width 172 after two x2 upsampling steps:
    # 43 -> 86 -> 172.
    assert h6.shape == (2, 32, 96, 172)

    print("\nModel block smoke test passed.")


if __name__ == "__main__":
    main()
