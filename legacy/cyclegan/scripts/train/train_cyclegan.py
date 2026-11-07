from __future__ import annotations

from pathlib import Path
import sys
import argparse
import json
import time
import subprocess
import signal
from dataclasses import asdict, dataclass
from contextlib import nullcontext
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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

from src.training.replay_buffer import ReplayBuffer


@dataclass
class TrainRunConfig:
    cache_root: str = "E:/Projects/Datasets/cache/cqt"
    output_root: str = "runs/cyclegan_cqt"

    experiment_name: str = "cqt96_baseline"

    sample_rate: int = 22050
    hop_length: int = 512
    n_bins: int = 96
    snippet_seconds: float = 4.0

    batch_size: int = 4
    num_workers: int = 0
    windows_per_track: int = 16
    min_window_energy: float = 0.01

    epochs: int = 50
    max_steps_per_epoch: int | None = None

    lr_g: float = 2e-4
    lr_d: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999

    grad_clip_norm: float | None = 5.0

    log_every: int = 25
    save_every_epochs: int = 5
    save_latest_every_steps: int = 500

    device: str = "cuda"

    amp: bool = False

    resume: str | None = None
    overfit_batches: int | None = None

    seed: int = 1337

    val_fraction: float = 0.1
    split_seed: int = 1337
    validate_every_epochs: int = 1
    val_batches: int = 20

    replay_buffer_size: int = 50
    replay_buffer_prob: float = 0.5

    preview_every_epochs: int = 1
    preview_every_steps: int | None = None
    preview_num_samples: int = 8
    preview_device: str = "cpu"
    disable_previews: bool = False


class TrainingControl:
    def __init__(self, control_dir: Path) -> None:
        self.control_dir = control_dir
        self.control_dir.mkdir(parents=True, exist_ok=True)

        self.pause_file = self.control_dir / "PAUSE"
        self.stop_file = self.control_dir / "STOP"
        self.save_file = self.control_dir / "SAVE_NOW"
        self.preview_file = self.control_dir / "PREVIEW_NOW"

        self.interrupted = False

    def request_stop_from_signal(self, signum=None, frame=None) -> None:
        print("\n[CONTROL] Interrupt received. Will save checkpoint and stop cleanly.")
        self.interrupted = True

    def should_stop(self) -> bool:
        return self.interrupted or self.stop_file.exists()

    def should_pause(self) -> bool:
        return self.pause_file.exists()

    def should_save_now(self) -> bool:
        return self.save_file.exists()

    def should_preview_now(self) -> bool:
        return self.preview_file.exists()

    def clear_save_now(self) -> None:
        if self.save_file.exists():
            self.save_file.unlink()

    def clear_preview_now(self) -> None:
        if self.preview_file.exists():
            self.preview_file.unlink()

    def wait_if_paused(self) -> None:
        if not self.should_pause():
            return

        print("[CONTROL] PAUSE file detected. Training paused.")
        print(f"[CONTROL] Remove this file to resume: {self.pause_file}")

        while self.should_pause() and not self.should_stop():
            time.sleep(5.0)

        print("[CONTROL] Pause released. Training resumed.")


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--cache-root", type=str, default="E:/Projects/Datasets/cache/cqt")
    parser.add_argument("--output-root", type=str, default="runs/cyclegan_cqt")
    parser.add_argument("--experiment-name", type=str, default="cqt96_baseline")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--windows-per-track", type=int, default=16)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--overfit-batches", type=int, default=None)

    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=1e-4)

    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--max-channels", type=int, default=256)
    parser.add_argument("--num-res-blocks", type=int, default=6)
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--residual-dropout", type=float, default=0.0)

    parser.add_argument("--lambda-cycle-x", type=float, default=10.0)
    parser.add_argument("--lambda-cycle-y", type=float, default=2.0)
    parser.add_argument("--lambda-identity-x", type=float, default=5.0)
    parser.add_argument("--lambda-identity-y", type=float, default=5.0)
    parser.add_argument("--lambda-chroma", type=float, default=2.0)
    parser.add_argument("--cycle-noise-std", type=float, default=0.03)
    parser.add_argument("--disable-cycle-noise", action="store_true")

    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every-epochs", type=int, default=5)
    parser.add_argument("--save-latest-every-steps", type=int, default=500)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--validate-every-epochs", type=int, default=1)
    parser.add_argument("--val-batches", type=int, default=20)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--replay-buffer-size", type=int, default=50)
    parser.add_argument("--replay-buffer-prob", type=float, default=0.5)
    parser.add_argument("--snippet-seconds", type=float, default=4.0)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)

    parser.add_argument("--preview-every-epochs", type=int, default=1)
    parser.add_argument("--preview-every-steps", type=int, default=None)
    parser.add_argument("--preview-num-samples", type=int, default=8)
    parser.add_argument("--preview-device", type=str, default="cpu")
    parser.add_argument("--disable-previews", action="store_true")

    return parser.parse_args()


def make_run_config(args: argparse.Namespace) -> TrainRunConfig:
    return TrainRunConfig(
        cache_root=args.cache_root,
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        windows_per_track=args.windows_per_track,
        epochs=args.epochs,
        max_steps_per_epoch=args.max_steps_per_epoch,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        log_every=args.log_every,
        save_every_epochs=args.save_every_epochs,
        save_latest_every_steps=args.save_latest_every_steps,
        device=args.device,
        resume=args.resume,
        overfit_batches=args.overfit_batches,
        seed=args.seed,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        validate_every_epochs=args.validate_every_epochs,
        val_batches=args.val_batches,
        snippet_seconds=args.snippet_seconds,
        grad_clip_norm=args.grad_clip_norm,
        amp=args.amp,
        replay_buffer_size=args.replay_buffer_size,
        replay_buffer_prob=args.replay_buffer_prob,
        preview_every_epochs=args.preview_every_epochs,
        preview_every_steps=args.preview_every_steps,
        preview_num_samples=args.preview_num_samples,
        preview_device=args.preview_device,
        disable_previews=args.disable_previews,
    )


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    models: CycleGANModels,
    optimizers: CycleGANOptimizers,
    train_config: TrainRunConfig,
    generator_config: GeneratorConfig,
    discriminator_config: DiscriminatorConfig,
    loss_config: CycleGANLossConfig,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "global_step": global_step,

        "g_x_to_y": models.g_x_to_y.state_dict(),
        "g_y_to_x": models.g_y_to_x.state_dict(),
        "d_x": models.d_x.state_dict(),
        "d_y": models.d_y.state_dict(),

        "opt_g": optimizers.g.state_dict(),
        "opt_d_x": optimizers.d_x.state_dict(),
        "opt_d_y": optimizers.d_y.state_dict(),

        "train_config": asdict(train_config),
        "generator_config": asdict(generator_config),
        "discriminator_config": asdict(discriminator_config),
        "loss_config": asdict(loss_config),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }

    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    models: CycleGANModels,
    optimizers: CycleGANOptimizers,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)

    models.g_x_to_y.load_state_dict(checkpoint["g_x_to_y"])
    models.g_y_to_x.load_state_dict(checkpoint["g_y_to_x"])
    models.d_x.load_state_dict(checkpoint["d_x"])
    models.d_y.load_state_dict(checkpoint["d_y"])

    optimizers.g.load_state_dict(checkpoint["opt_g"])
    optimizers.d_x.load_state_dict(checkpoint["opt_d_x"])
    optimizers.d_y.load_state_dict(checkpoint["opt_d_y"])

    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = int(checkpoint["epoch"]) + 1
    global_step = int(checkpoint["global_step"])

    return start_epoch, global_step


def log_losses(writer: SummaryWriter, losses: dict[str, float], global_step: int) -> None:
    for key, value in losses.items():
        writer.add_scalar(key, value, global_step)


def run_preview_export(
    checkpoint_path: Path,
    train_config: TrainRunConfig,
    run_dir: Path,
    epoch: int,
    global_step: int,
) -> None:
    if train_config.disable_previews:
        return

    preview_dir = (
        run_dir
        / "previews"
        / f"step_{global_step:09d}_epoch_{epoch + 1:04d}"
    )

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval" / "export_cyclegan_preview.py"),

        "--checkpoint",
        str(checkpoint_path),

        "--cache-root",
        train_config.cache_root,

        "--out-dir",
        str(preview_dir),

        "--split",
        "val",

        "--val-fraction",
        str(train_config.val_fraction),

        "--split-seed",
        str(train_config.split_seed),

        "--num-samples",
        str(train_config.preview_num_samples),

        "--snippet-seconds",
        str(train_config.snippet_seconds),

        "--sample-rate",
        str(train_config.sample_rate),

        "--hop-length",
        str(train_config.hop_length),

        "--device",
        train_config.preview_device,

        "--save-tensors",
    ]

    print("\nRunning automatic preview export:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))

    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    except Exception as exc:
        # Do not kill a long training run just because preview export failed.
        print(f"[WARNING] Preview export failed: {exc}")


def save_training_state(
    checkpoint_dir: Path,
    epoch: int,
    global_step: int,
    models: CycleGANModels,
    optimizers: CycleGANOptimizers,
    train_cfg: TrainRunConfig,
    generator_cfg: GeneratorConfig,
    discriminator_cfg: DiscriminatorConfig,
    loss_cfg: CycleGANLossConfig,
    scaler,
    checkpoint_name: str = "latest.pt",
) -> Path:
    path = checkpoint_dir / checkpoint_name

    save_checkpoint(
        path=path,
        epoch=epoch,
        global_step=global_step,
        models=models,
        optimizers=optimizers,
        train_config=train_cfg,
        generator_config=generator_cfg,
        discriminator_config=discriminator_cfg,
        loss_config=loss_cfg,
        scaler=scaler,
    )

    return path


@torch.no_grad()
def run_validation(
    val_loader,
    models: CycleGANModels,
    loss_bundle,
    device: torch.device,
    max_batches: int = 20,
    use_amp: bool = False,
) -> dict[str, float]:
    from vocoder.src.nes_vocoder.models.src.losses.cyclegan_losses import (
        compute_generator_losses,
        compute_discriminator_loss,
        inject_gaussian_noise,
    )

    models.g_x_to_y.eval()
    models.g_y_to_x.eval()
    models.d_x.eval()
    models.d_y.eval()

    cfg = loss_bundle.config

    totals: dict[str, float] = {}
    count = 0

    amp_enabled = bool(use_amp and device.type == "cuda")
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if amp_enabled
        else nullcontext()
    )

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        real_x = batch["real_x"].to(device, non_blocking=True)
        real_y = batch["real_y"].to(device, non_blocking=True)

        with autocast_ctx:
            fake_y = models.g_x_to_y(real_x)
            fake_x = models.g_y_to_x(real_y)

            noisy_fake_y = inject_gaussian_noise(
                fake_y,
                std=cfg.cycle_noise_std,
                enabled=False,
            )
            noisy_fake_x = inject_gaussian_noise(
                fake_x,
                std=cfg.cycle_noise_std,
                enabled=False,
            )

            rec_x = models.g_y_to_x(noisy_fake_y)
            rec_y = models.g_x_to_y(noisy_fake_x)

            id_x = models.g_y_to_x(real_x)
            id_y = models.g_x_to_y(real_y)

            pred_fake_y_for_g = models.d_y(fake_y)
            pred_fake_x_for_g = models.d_x(fake_x)

            g_losses = compute_generator_losses(
                real_x=real_x,
                real_y=real_y,
                fake_y=fake_y,
                fake_x=fake_x,
                rec_x=rec_x,
                rec_y=rec_y,
                id_x=id_x,
                id_y=id_y,
                pred_fake_y=pred_fake_y_for_g,
                pred_fake_x=pred_fake_x_for_g,
                bundle=loss_bundle,
            )

            d_x_losses = compute_discriminator_loss(
                pred_real=models.d_x(real_x),
                pred_fake_detached=models.d_x(fake_x.detach()),
                bundle=loss_bundle,
            )

            d_y_losses = compute_discriminator_loss(
                pred_real=models.d_y(real_y),
                pred_fake_detached=models.d_y(fake_y.detach()),
                bundle=loss_bundle,
            )

            values = {
                "val_loss_g_total": g_losses.total,
                "val_loss_g_cycle_x": g_losses.cycle_x,
                "val_loss_g_cycle_y": g_losses.cycle_y,
                "val_loss_g_chroma": g_losses.chroma,
                "val_loss_d_x_total": d_x_losses.total,
                "val_loss_d_y_total": d_y_losses.total,
            }

            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu().item())

            count += 1

    if count == 0:
        return {}

    return {
        key: value / count
        for key, value in totals.items()
    }


def main() -> None:
    args = parse_args()
    train_cfg = make_run_config(args)

    set_seed(train_cfg.seed)

    if train_cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(train_cfg.device)

    run_dir = Path(train_cfg.output_root) / train_cfg.experiment_name

    control = TrainingControl(run_dir / "control")

    signal.signal(signal.SIGINT, control.request_stop_from_signal)
    signal.signal(signal.SIGTERM, control.request_stop_from_signal)

    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "tensorboard"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    generator_cfg = GeneratorConfig(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        max_channels=args.max_channels,
        num_downsamples=2,
        num_res_blocks=args.num_res_blocks,
        residual_dropout=args.residual_dropout,
        use_attention=not args.no_attention,
        attention_position="middle",
        norm="instance",
        padding_mode="reflect",
    )

    discriminator_cfg = DiscriminatorConfig(
        in_channels=1,
        base_channels=args.base_channels,
        max_channels=512,
        num_layers=3,
        norm="instance",
        spectral_norm=False,
        use_sigmoid=False,
    )

    loss_cfg = CycleGANLossConfig(
        lambda_cycle_x=args.lambda_cycle_x,
        lambda_cycle_y=args.lambda_cycle_y,
        lambda_identity_x=args.lambda_identity_x,
        lambda_identity_y=args.lambda_identity_y,
        lambda_chroma=args.lambda_chroma,
        cycle_noise_std=args.cycle_noise_std,
        cycle_noise_enabled=not args.disable_cycle_noise,
    )

    save_json(
        run_dir / "config.json",
        {
            "train_config": asdict(train_cfg),
            "generator_config": asdict(generator_cfg),
            "discriminator_config": asdict(discriminator_cfg),
            "loss_config": asdict(loss_cfg),
        },
    )

    print("Building dataloaders...")

    train_loader = build_unpaired_cqt_dataloader(
        cache_root=Path(train_cfg.cache_root),
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        shuffle=True,
        snippet_seconds=train_cfg.snippet_seconds,
        sample_rate=train_cfg.sample_rate,
        hop_length=train_cfg.hop_length,
        windows_per_track=train_cfg.windows_per_track,
        return_chroma=True,
        return_phase=False,
        return_metadata=False,
        min_window_energy=train_cfg.min_window_energy,
        split="train",
        val_fraction=train_cfg.val_fraction,
        split_seed=train_cfg.split_seed,
    )

    val_loader = build_unpaired_cqt_dataloader(
        cache_root=Path(train_cfg.cache_root),
        batch_size=train_cfg.batch_size,
        num_workers=0,
        shuffle=False,
        snippet_seconds=train_cfg.snippet_seconds,
        sample_rate=train_cfg.sample_rate,
        hop_length=train_cfg.hop_length,
        windows_per_track=2,
        return_chroma=True,
        return_phase=False,
        return_metadata=False,
        min_window_energy=train_cfg.min_window_energy,
        split="val",
        val_fraction=train_cfg.val_fraction,
        split_seed=train_cfg.split_seed,
    )

    if train_cfg.overfit_batches is not None:
        print(f"Overfit mode enabled: using first {train_cfg.overfit_batches} batches repeatedly.")
        overfit_batches = []
        iterator = iter(train_loader)
        for _ in range(train_cfg.overfit_batches):
            overfit_batches.append(next(iterator))
    else:
        overfit_batches = None

    print("Building models...")

    g_x_to_y = AudioResnetGenerator(generator_cfg).to(device)
    g_y_to_x = AudioResnetGenerator(generator_cfg).to(device)
    d_x = PatchGANDiscriminator(discriminator_cfg).to(device)
    d_y = PatchGANDiscriminator(discriminator_cfg).to(device)

    models = CycleGANModels(
        g_x_to_y=g_x_to_y,
        g_y_to_x=g_y_to_x,
        d_x=d_x,
        d_y=d_y,
    )

    optimizers = CycleGANOptimizers(
        g=torch.optim.Adam(
            list(g_x_to_y.parameters()) + list(g_y_to_x.parameters()),
            lr=train_cfg.lr_g,
            betas=(train_cfg.beta1, train_cfg.beta2),
        ),
        d_x=torch.optim.Adam(
            d_x.parameters(),
            lr=train_cfg.lr_d,
            betas=(train_cfg.beta1, train_cfg.beta2),
        ),
        d_y=torch.optim.Adam(
            d_y.parameters(),
            lr=train_cfg.lr_d,
            betas=(train_cfg.beta1, train_cfg.beta2),
        ),
    )

    loss_bundle = build_cyclegan_loss_bundle(loss_cfg)

    scaler = torch.amp.GradScaler("cuda",
                                  enabled=(train_cfg.amp and device.type == "cuda"))

    fake_x_buffer = ReplayBuffer(
        max_size=train_cfg.replay_buffer_size,
        return_old_probability=train_cfg.replay_buffer_prob,
    )
    fake_y_buffer = ReplayBuffer(
        max_size=train_cfg.replay_buffer_size,
        return_old_probability=train_cfg.replay_buffer_prob,
    )

    start_epoch = 0
    global_step = 0

    if train_cfg.resume is not None:
        print(f"Resuming from: {train_cfg.resume}")
        start_epoch, global_step = load_checkpoint(
            path=Path(train_cfg.resume),
            models=models,
            optimizers=optimizers,
            device=device,
            scaler=scaler,
        )
        print(f"Resumed at epoch={start_epoch}, global_step={global_step}")

    writer = SummaryWriter(log_dir=str(log_dir))

    print("Starting training.")
    print(f"Run dir: {run_dir}")
    print(f"Device: {device}")
    print(f"Batches per epoch: {len(train_loader)}")

    for epoch in range(start_epoch, train_cfg.epochs):
        epoch_start = time.time()

        g_x_to_y.train()
        g_y_to_x.train()
        d_x.train()
        d_y.train()

        if overfit_batches is not None:
            epoch_iterable = overfit_batches
        else:
            epoch_iterable = train_loader

        progress = tqdm(
            epoch_iterable,
            desc=f"Epoch {epoch + 1}/{train_cfg.epochs}",
            leave=True,
        )

        running_losses: dict[str, float] = {}

        for step_in_epoch, batch in enumerate(progress):
            if (
                train_cfg.max_steps_per_epoch is not None
                and step_in_epoch >= train_cfg.max_steps_per_epoch
            ):
                break

            losses = cyclegan_train_step(
                batch=batch,
                models=models,
                optimizers=optimizers,
                loss_bundle=loss_bundle,
                device=device,
                grad_clip_norm=train_cfg.grad_clip_norm,
                use_amp=train_cfg.amp,
                scaler=scaler,
                fake_x_buffer=fake_x_buffer,
                fake_y_buffer=fake_y_buffer,
            )

            global_step += 1

            for key, value in losses.items():
                running_losses[key] = value

            if global_step % train_cfg.log_every == 0:
                log_losses(writer, losses, global_step)

            progress.set_postfix(
                {
                    "G": f"{losses['loss_g_total']:.3f}",
                    "Dx": f"{losses['loss_d_x_total']:.3f}",
                    "Dy": f"{losses['loss_d_y_total']:.3f}",
                    "cycX": f"{losses['loss_g_cycle_x']:.3f}",
                    "cycY": f"{losses['loss_g_cycle_y']:.3f}",
                }
            )

            if global_step % train_cfg.save_latest_every_steps == 0:
                save_checkpoint(
                    path=checkpoint_dir / "latest.pt",
                    epoch=epoch,
                    global_step=global_step,
                    models=models,
                    optimizers=optimizers,
                    train_config=train_cfg,
                    generator_config=generator_cfg,
                    discriminator_config=discriminator_cfg,
                    loss_config=loss_cfg,
                    scaler=scaler,
                )

            if (
                    train_cfg.preview_every_steps is not None
                    and train_cfg.preview_every_steps > 0
                    and global_step % train_cfg.preview_every_steps == 0
            ):
                step_checkpoint_path = checkpoint_dir / f"step_{global_step:09d}.pt"

                save_checkpoint(
                    path=step_checkpoint_path,
                    epoch=epoch,
                    global_step=global_step,
                    models=models,
                    optimizers=optimizers,
                    train_config=train_cfg,
                    generator_config=generator_cfg,
                    discriminator_config=discriminator_cfg,
                    loss_config=loss_cfg,
                    scaler=scaler,
                )

                run_preview_export(
                    checkpoint_path=step_checkpoint_path,
                    train_config=train_cfg,
                    run_dir=run_dir,
                    epoch=epoch,
                    global_step=global_step,
                )

            control.wait_if_paused()

            if control.should_save_now():
                manual_checkpoint = save_training_state(
                    checkpoint_dir=checkpoint_dir,
                    epoch=epoch,
                    global_step=global_step,
                    models=models,
                    optimizers=optimizers,
                    train_cfg=train_cfg,
                    generator_cfg=generator_cfg,
                    discriminator_cfg=discriminator_cfg,
                    loss_cfg=loss_cfg,
                    scaler=scaler,
                    checkpoint_name=f"manual_step_{global_step:09d}.pt",
                )

                print(
                    f"[CONTROL] Manual checkpoint saved: {manual_checkpoint}")
                control.clear_save_now()

            if control.should_preview_now():
                preview_checkpoint = save_training_state(
                    checkpoint_dir=checkpoint_dir,
                    epoch=epoch,
                    global_step=global_step,
                    models=models,
                    optimizers=optimizers,
                    train_cfg=train_cfg,
                    generator_cfg=generator_cfg,
                    discriminator_cfg=discriminator_cfg,
                    loss_cfg=loss_cfg,
                    scaler=scaler,
                    checkpoint_name=f"preview_step_{global_step:09d}.pt",
                )

                run_preview_export(
                    checkpoint_path=preview_checkpoint,
                    train_config=train_cfg,
                    run_dir=run_dir,
                    epoch=epoch,
                    global_step=global_step,
                )

                control.clear_preview_now()

            if control.should_stop():
                stop_checkpoint = save_training_state(
                    checkpoint_dir=checkpoint_dir,
                    epoch=epoch,
                    global_step=global_step,
                    models=models,
                    optimizers=optimizers,
                    train_cfg=train_cfg,
                    generator_cfg=generator_cfg,
                    discriminator_cfg=discriminator_cfg,
                    loss_cfg=loss_cfg,
                    scaler=scaler,
                    checkpoint_name=f"stop_step_{global_step:09d}.pt",
                )

                save_training_state(
                    checkpoint_dir=checkpoint_dir,
                    epoch=epoch,
                    global_step=global_step,
                    models=models,
                    optimizers=optimizers,
                    train_cfg=train_cfg,
                    generator_cfg=generator_cfg,
                    discriminator_cfg=discriminator_cfg,
                    loss_cfg=loss_cfg,
                    scaler=scaler,
                    checkpoint_name="latest.pt",
                )

                print(f"[CONTROL] Stop requested. Saved: {stop_checkpoint}")
                writer.close()
                return

        epoch_seconds = time.time() - epoch_start

        writer.add_scalar("epoch/duration_seconds", epoch_seconds, epoch + 1)

        if running_losses:
            for key, value in running_losses.items():
                writer.add_scalar(f"epoch_last/{key}", value, epoch + 1)

        print(f"Epoch {epoch + 1} finished in {epoch_seconds:.1f}s")

        save_checkpoint(
            path=checkpoint_dir / "latest.pt",
            epoch=epoch,
            global_step=global_step,
            models=models,
            optimizers=optimizers,
            train_config=train_cfg,
            generator_config=generator_cfg,
            discriminator_config=discriminator_cfg,
            loss_config=loss_cfg,
            scaler=scaler,
        )

        if (epoch + 1) % train_cfg.save_every_epochs == 0:
            save_checkpoint(
                path=checkpoint_dir / f"epoch_{epoch + 1:04d}.pt",
                epoch=epoch,
                global_step=global_step,
                models=models,
                optimizers=optimizers,
                train_config=train_cfg,
                generator_config=generator_cfg,
                discriminator_config=discriminator_cfg,
                loss_config=loss_cfg,
                scaler=scaler,
            )

        if (
                not train_cfg.disable_previews
                and train_cfg.preview_every_epochs > 0
                and (epoch + 1) % train_cfg.preview_every_epochs == 0
        ):
            preview_checkpoint_path = checkpoint_dir / f"step_{global_step:09d}.pt"

            save_checkpoint(
                path=preview_checkpoint_path,
                epoch=epoch,
                global_step=global_step,
                models=models,
                optimizers=optimizers,
                train_config=train_cfg,
                generator_config=generator_cfg,
                discriminator_config=discriminator_cfg,
                loss_config=loss_cfg,
                scaler=scaler,
            )

            run_preview_export(
                checkpoint_path=preview_checkpoint_path,
                train_config=train_cfg,
                run_dir=run_dir,
                epoch=epoch,
                global_step=global_step,

            )

        if (epoch + 1) % train_cfg.validate_every_epochs == 0:
            val_losses = run_validation(
                val_loader=val_loader,
                models=models,
                loss_bundle=loss_bundle,
                device=device,
                max_batches=train_cfg.val_batches,
                use_amp=train_cfg.amp,
            )

            if val_losses:
                print("Validation:")
                for key, value in sorted(val_losses.items()):
                    print(f"  {key}: {value:.6f}")
                    writer.add_scalar(key, value, epoch + 1)

    writer.close()

    print("Training complete.")


if __name__ == "__main__":
    main()

