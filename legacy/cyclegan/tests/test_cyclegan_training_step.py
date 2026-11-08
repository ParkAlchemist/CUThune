import torch

from legacy.cyclegan.losses.cyclegan_losses import build_cyclegan_loss_bundle, CycleGANLossConfig
from legacy.cyclegan.models.cyclegan import GeneratorConfig, DiscriminatorConfig, AudioResnetGenerator, \
    PatchGANDiscriminator
from legacy.cyclegan.src.cyclegan_step import CycleGANModels, CycleGANOptimizers, cyclegan_train_step
from legacy.cyclegan.src.replay_buffer import ReplayBuffer
from tests.test_config import CACHE_ROOT

from cut.src.nes_cut.data.unpaired_dataset import build_unpaired_cqt_dataloader

def clone_first_parameter(model: torch.nn.Module) -> torch.Tensor:
    for param in model.parameters():
        if param.requires_grad:
            return param.detach().clone()
    raise RuntimeError("Model has no trainable parameters")


def first_parameter_changed(model: torch.nn.Module, before: torch.Tensor) -> bool:
    for param in model.parameters():
        if param.requires_grad:
            return not torch.allclose(param.detach(), before)
    return False


def test_one_cyclegan_training_step_cpu():
    device = torch.device("cpu")

    loader = build_unpaired_cqt_dataloader(
        cache_root=CACHE_ROOT,
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
        base_channels=8,
        max_channels=64,
        num_res_blocks=1,
        use_attention=False,
    )

    disc_cfg = DiscriminatorConfig(
        base_channels=8,
        max_channels=64,
        num_layers=2,
    )

    g_x_to_y = AudioResnetGenerator(gen_cfg).to(device)
    g_y_to_x = AudioResnetGenerator(gen_cfg).to(device)
    d_x = PatchGANDiscriminator(disc_cfg).to(device)
    d_y = PatchGANDiscriminator(disc_cfg).to(device)

    before_g = clone_first_parameter(g_x_to_y)
    before_d = clone_first_parameter(d_x)

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

    loss_bundle = build_cyclegan_loss_bundle(
        CycleGANLossConfig(
            lambda_chroma=1.0,
            cycle_noise_enabled=True,
        )
    )

    losses = cyclegan_train_step(
        batch=batch,
        models=models,
        optimizers=optimizers,
        loss_bundle=loss_bundle,
        device=device,
        grad_clip_norm=5.0,
        fake_x_buffer=ReplayBuffer(max_size=5),
        fake_y_buffer=ReplayBuffer(max_size=5),
        use_amp=False,
        scaler=None,
    )

    assert losses["loss_g_total"] > 0
    assert losses["loss_d_x_total"] > 0
    assert losses["loss_d_y_total"] > 0

    for key, value in losses.items():
        assert torch.isfinite(torch.tensor(value)), f"Non-finite value: {key}={value}"

    assert first_parameter_changed(g_x_to_y, before_g)
    assert first_parameter_changed(d_x, before_d)

