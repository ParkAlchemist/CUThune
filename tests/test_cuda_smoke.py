import pytest
import torch

from legacy.cyclegan.models.cyclegan import AudioResnetGenerator, GeneratorConfig, PatchGANDiscriminator, \
    DiscriminatorConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_generator_and_discriminator_cuda_forward():
    device = torch.device("cuda")

    x = torch.randn(2, 1, 96, 172, device=device)

    generator = AudioResnetGenerator(
        GeneratorConfig(
            base_channels=32,
            num_res_blocks=6,
            use_attention=True,
        )
    ).to(device)

    discriminator = PatchGANDiscriminator(
        DiscriminatorConfig(
            base_channels=32,
            num_layers=3,
        )
    ).to(device)

    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        fake = generator(x)
        pred = discriminator(fake)

    assert fake.shape == x.shape
    assert pred.ndim == 4
    assert pred.shape[0] == x.shape[0]
    assert pred.shape[1] == 1

