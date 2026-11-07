from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from cut.src.nes_cut.data.unpaired_dataset import build_unpaired_cqt_dataloader
from src.models.cyclegan import (
    GeneratorConfig,
    DiscriminatorConfig,
    AudioResnetGenerator,
    PatchGANDiscriminator,
)
from vocoder.src.nes_vocoder.models.src.losses.cyclegan_losses import (
    CycleGANLossConfig,
    build_cyclegan_loss_bundle,
)
from vocoder.src.nes_vocoder.models.src.training.cyclegan_step import (
    CycleGANModels,
    CycleGANOptimizers,
    cyclegan_train_step,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_root = Path("E:/Projects/Datasets/cache/cqt")

    loader = build_unpaired_cqt_dataloader(
        cache_root=cache_root,
        batch_size=2,
        num_workers=0,
        shuffle=True,
        snippet_seconds=4.0,
        sample_rate=22050,
        hop_length=512,
        windows_per_track=2,
        return_chroma=True,
        return_phase=False,
        return_metadata=False,
        min_window_energy=0.01,
    )

    batch = next(iter(loader))

    gen_cfg = GeneratorConfig(
        base_channels=16,
        max_channels=128,
        num_res_blocks=2,
        use_attention=False,
        residual_dropout=0.0,
    )

    disc_cfg = DiscriminatorConfig(
        base_channels=16,
        max_channels=128,
        num_layers=3,
        spectral_norm=False,
    )

    g_x_to_y = AudioResnetGenerator(gen_cfg).to(device)
    g_y_to_x = AudioResnetGenerator(gen_cfg).to(device)
    d_x = PatchGANDiscriminator(disc_cfg).to(device)
    d_y = PatchGANDiscriminator(disc_cfg).to(device)

    models = CycleGANModels(
        g_x_to_y=g_x_to_y,
        g_y_to_x=g_y_to_x,
        d_x=d_x,
        d_y=d_y,
    )

    optimizers = CycleGANOptimizers(
        g=torch.optim.Adam(
            list(g_x_to_y.parameters()) + list(g_y_to_x.parameters()),
            lr=2e-4,
            betas=(0.5, 0.999),
        ),
        d_x=torch.optim.Adam(
            d_x.parameters(),
            lr=1e-4,
            betas=(0.5, 0.999),
        ),
        d_y=torch.optim.Adam(
            d_y.parameters(),
            lr=1e-4,
            betas=(0.5, 0.999),
        ),
    )

    loss_cfg = CycleGANLossConfig(
        lambda_cycle_x=10.0,
        lambda_cycle_y=2.0,
        lambda_identity_x=5.0,
        lambda_identity_y=5.0,
        lambda_chroma=2.0,
        cycle_noise_std=0.03,
        cycle_noise_enabled=True,
    )

    loss_bundle = build_cyclegan_loss_bundle(loss_cfg)

    losses = cyclegan_train_step(
        batch=batch,
        models=models,
        optimizers=optimizers,
        loss_bundle=loss_bundle,
        device=device,
        grad_clip_norm=5.0,
    )

    print("Device:", device)
    print("Losses:")
    for key, value in sorted(losses.items()):
        print(f"  {key}: {value:.6f}")

    for key, value in losses.items():
        assert torch.isfinite(torch.tensor(value)), f"Non-finite loss: {key}={value}"

    print("\nCycleGAN one-step training smoke test passed.")


if __name__ == "__main__":
    main()

