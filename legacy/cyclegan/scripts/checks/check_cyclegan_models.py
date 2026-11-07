from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models.cyclegan import (
    GeneratorConfig,
    DiscriminatorConfig,
    AudioResnetGenerator,
    PatchGANDiscriminator,
    MultiScalePatchGANDiscriminator,
    describe_model,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(2, 1, 96, 172, device=device)

    gen_config = GeneratorConfig(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        max_channels=256,
        num_downsamples=2,
        num_res_blocks=6,
        residual_dropout=0.0,
        use_attention=True,
        attention_position="middle",
        norm="instance",
        padding_mode="reflect",
    )

    disc_config = DiscriminatorConfig(
        in_channels=1,
        base_channels=32,
        max_channels=512,
        num_layers=3,
        norm="instance",
        spectral_norm=False,
        use_sigmoid=False,
    )

    generator = AudioResnetGenerator(gen_config).to(device)
    discriminator = PatchGANDiscriminator(disc_config).to(device)
    multiscale_discriminator = MultiScalePatchGANDiscriminator(
        config=disc_config,
        num_scales=2,
    ).to(device)

    generator.eval()
    discriminator.eval()
    multiscale_discriminator.eval()

    with torch.no_grad():
        y = generator(x)
        d_real = discriminator(x)
        d_fake = discriminator(y)
        d_multi = multiscale_discriminator(x)

    print("Device:", device)
    print(describe_model(generator))
    print(describe_model(discriminator))
    print(describe_model(multiscale_discriminator))

    print("\nShapes:")
    print("  input:", tuple(x.shape))
    print("  generator output:", tuple(y.shape))
    print("  discriminator real:", tuple(d_real.shape))
    print("  discriminator fake:", tuple(d_fake.shape))

    for i, out in enumerate(d_multi):
        print(f"  multiscale[{i}]:", tuple(out.shape))

    print("\nValue range:")
    print("  generator min:", float(y.min().item()))
    print("  generator max:", float(y.max().item()))

    assert y.shape == x.shape
    assert d_real.ndim == 4
    assert d_real.shape[0] == x.shape[0]
    assert d_real.shape[1] == 1
    assert d_fake.shape == d_real.shape
    assert len(d_multi) == 2

    print("\nCycleGAN model smoke test passed.")


if __name__ == "__main__":
    main()

