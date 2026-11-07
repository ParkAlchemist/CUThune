from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Iterable
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch

from src.training.vocoder.accumulation import (
    average_loss_dicts,
    accumulation_windows,
)
from src.training.vocoder.context import (
    TrainingComponents,
    TrainingState,
)
from src.training.vocoder.runtime import derive_training_seed, set_seed
from src.training.vocoder_step import (
    finish_vocoder_optimizer_step,
    vocoder_train_micro_step, zero_vocoder_gradients,
)
from src.training.vocoder.schedulers import (
    step_vocoder_schedulers, get_vocoder_learning_rates,
)
from src.training.vocoder.checkpointing import (
    prune_numbered_checkpoints,
    save_checkpoint,
)
from src.training.vocoder.preview import (
    export_preview_wavs,
)
from src.training.vocoder.logging import (
    collect_context_parameter_metrics,
    write_status, )
from src.training.vocoder.metrics import split_logging_metrics, log_interval_maxima


@dataclass
class OptimizerUpdateResult:
    losses: dict[str, float]
    samples_processed: int
    microbatches_processed: int


def amp_dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16

    if name == "bfloat16":
        return torch.bfloat16

    raise ValueError(
        f"Unsupported AMP dtype: {name!r}"
    )


def perform_optimizer_update(
    *,
    microbatches: list[dict],
    config,
    components: TrainingComponents,
    state: TrainingState,
    device: torch.device,
) -> OptimizerUpdateResult:
    if not microbatches:
        raise ValueError(
            "Cannot perform an optimizer update with no batches."
        )

    next_global_step = state.global_step + 1

    use_amp = bool(
        config.amp.enabled
        and device.type == "cuda"
        and next_global_step >= config.amp.start_step
    )

    accumulation_size = len(microbatches)

    optimizers = components.optimizers

    zero_vocoder_gradients(optimizers)

    state.accumulation_in_progress = True

    micro_losses: list[dict[str, float]] = []
    samples_processed = 0

    try:
        for batch in microbatches:
            result = vocoder_train_micro_step(
                batch=batch,
                models=components.models,
                loss_bundle=components.loss_bundle,
                device=device,
                use_amp=use_amp,
                scaler=components.scaler if use_amp else None,
                dtype=config.amp.dtype,
                loss_divisor=accumulation_size,
            )

            if not all(
                np.isfinite(value)
                for value in result.losses.values()
            ):
                raise FloatingPointError(
                    "Non-finite micro-step loss before optimizer "
                    f"update {next_global_step}: {result.losses}"
                )

            micro_losses.append(result.losses)
            samples_processed += result.batch_size

        losses = average_loss_dicts(micro_losses)

        grad_metrics = finish_vocoder_optimizer_step(
            models=components.models,
            optimizers=optimizers,
            scaler=(
                components.scaler
                if use_amp
                else None
            ),
            use_amp=use_amp,
            grad_clip_generator=(
                config.training.grad_clip_generator
            ),
            grad_clip_discriminator=(
                config.training.grad_clip_discriminator
            ),
        )

        step_vocoder_schedulers(
            components=components,
            generator_config=config.scheduler_generator,
            discriminator_config=config.scheduler_discriminator,
            interval="optimizer_step"
        )

    except Exception:
        optimizers.generator.zero_grad(set_to_none=True)
        optimizers.discriminator.zero_grad(set_to_none=True)
        raise

    finally:
        state.accumulation_in_progress = False

    for key, value in grad_metrics.items():
        if value is not None:
            losses[key] = float(value)

    state.global_step = next_global_step
    state.micro_step += accumulation_size
    state.segments_seen += samples_processed
    state.audio_samples_seen += (
        samples_processed
        * config.data.segment_frames
        * config.data.hop_length
    )

    losses["samples_in_update"] = float(
        samples_processed
    )
    losses["microbatches_in_update"] = float(
        accumulation_size
    )
    losses["use_amp"] = float(use_amp)

    losses.update(get_vocoder_learning_rates(components))

    return OptimizerUpdateResult(
        losses=losses,
        samples_processed=samples_processed,
        microbatches_processed=accumulation_size,
    )


def should_run(
    step: int,
    interval: int,
) -> bool:
    return interval > 0 and step % interval == 0


def save_scheduled_checkpoints(
    *,
    config,
    paths,
    components,
    state,
) -> None:
    step_path = (
        paths.checkpoint_dir
        / f"step_{state.global_step:09d}.pt"
    )

    save_checkpoint(
        step_path,
        state=state,
        components=components,
        config=config,
    )

    print(f"\nSaved checkpoint: {step_path}")

    prune_numbered_checkpoints(
        paths.checkpoint_dir,
        config.logging.keep_numbered_checkpoints,
    )

    latest_path = paths.checkpoint_dir / "latest.pt"

    save_checkpoint(
        latest_path,
        state=state,
        components=components,
        config=config,
    )
    state.latest_checkpoint = str(latest_path)


def run_epoch(
    *,
    epoch: int,
    runtime,
    state: TrainingState,
    resume_seed_offset: int = 0,
) -> None:

    epoch_seed = derive_training_seed(
        base_seed=runtime.config.run.seed,
        epoch=epoch,
        resume_global_step=resume_seed_offset,
    )

    set_seed(epoch_seed)

    print(f"Epoch: {epoch + 1}, random seed: {epoch_seed}")

    if runtime.writer is not None:
        runtime.writer.add_scalar(
            "run/epoch_seed",
            epoch_seed,
            state.global_step,
        )

    config = runtime.config
    components = runtime.components

    components.models.generator.train()
    components.models.discriminator.train()
    epoch_iterable = (
        runtime.overfit_batches
        if runtime.overfit_batches is not None
        else runtime.train_loader
    )

    accumulation_steps = (
        config.training.gradient_accumulation_steps
    )

    number_of_microbatches = len(epoch_iterable)
    number_of_updates = math.ceil(
        number_of_microbatches
        / accumulation_steps
    )

    progress = tqdm(
        accumulation_windows(
            epoch_iterable,
            accumulation_steps,
        ),
        total=number_of_updates,
        desc=(
            f"Epoch {epoch + 1}/"
            f"{config.training.epochs}"
        ),
        leave=True,
    )

    for microbatches in progress:
        state.epoch = epoch

        try:
            update = perform_optimizer_update(
                microbatches=microbatches,
                config=config,
                components=components,
                state=state,
                device=runtime.device,
            )

            step_metrics = dict(update.losses)

            step_metrics.update(
                collect_context_parameter_metrics(
                    components.models.generator
                )
            )

            step_metrics.update(
                get_vocoder_learning_rates(
                    components
                )
            )

            averaged_metrics, latest_metrics = split_logging_metrics(step_metrics)

            runtime.metric_accumulator.update(averaged_metrics)

        except FloatingPointError as exc:
            emergency_path = (
                runtime.paths.checkpoint_dir
                / (
                    "nonfinite_before_step_"
                    f"{state.global_step + 1:09d}.pt"
                )
            )

            save_checkpoint(
                emergency_path,
                state=state,
                components=components,
                config=config,
            )

            message = f"{exc}. Saved {emergency_path}"

            if config.training.fail_on_nonfinite:
                raise RuntimeError(message) from exc

            print(f"\nWARNING: {message}")
            continue

        losses = update.losses
        step = state.global_step

        if (
                runtime.writer is not None
                and should_run(
            step,
            config.logging.log_every_steps,
        )
        ):
            interval_means = runtime.metric_accumulator.means()

            for key, value in interval_means.items():
                runtime.writer.add_scalar(
                    key,
                    value,
                    step,
                )

            for name, value in latest_metrics.items():
                runtime.writer.add_scalar(
                    name,
                    value,
                    step,
                )

            runtime.writer.add_scalar(
                "logging/window_count",
                runtime.metric_accumulator.maximum_count(),
                step,
            )

            log_interval_maxima(
                writer=runtime.writer,
                accumulator=runtime.metric_accumulator,
                global_step=step,
            )

            runtime.metric_accumulator.reset()

        progress.set_postfix(
            {
                "G": f"{losses['loss_g_total']:.2f}",
                "D": f"{losses['loss_d_total']:.3f}",
                "MR": f"{losses['loss_g_mrstft']:.3f}",
            }
        )

        if should_run(
            step,
            config.logging.save_every_steps,
        ):
            save_scheduled_checkpoints(
                config=config,
                paths=runtime.paths,
                components=components,
                state=state,
            )

        if should_run(
            step,
            config.logging.preview_every_steps,
        ):
            preview_path = (
                runtime.paths.preview_dir
                / f"step_{step:09d}"
            )

            preview_batches = (
                runtime.overfit_batches
                if runtime.overfit_batches is not None
                else runtime.preview_loader
            )

            export_preview_wavs(
                preview_dir=preview_path,
                generator=components.models.generator,
                preview_batches=preview_batches,
                device=runtime.device,
                sample_rate=config.data.sample_rate,
                num_samples=(
                    config.logging.preview_num_samples
                ),
                use_amp=bool(losses["use_amp"]),
                amp_dtype=amp_dtype_from_name(
                    config.amp.dtype
                ),
                writer=runtime.writer,
                global_step=step,
                hop_length=config.data.hop_length,
                cqt_bins=config.data.cqt_bins,
                bins_per_octave=config.data.bins_per_octave,
                fmin=config.data.fmin,
            )

            state.latest_preview = str(preview_path)

            print(
                f"\nSaved preview WAVs: {preview_path}"
            )

        if should_run(
            step,
            config.logging.status_every_steps,
        ):
            learning_rates = get_vocoder_learning_rates(runtime.components)

            write_status(
                paths=runtime.paths,
                state=state,
                losses={**losses, **learning_rates},
            )

        if (
            config.training.max_steps is not None
            and step >= config.training.max_steps
        ):
            print("Reached max_steps.")
            runtime.stop_controller.stop_requested = True

        if runtime.stop_controller.stop_requested:
            raise KeyboardInterrupt


def save_latest(
    runtime,
    state: TrainingState,
) -> Path:
    latest_path = (
        runtime.paths.checkpoint_dir
        / "latest.pt"
    )

    save_checkpoint(
        latest_path,
        state=state,
        components=runtime.components,
        config=runtime.config,
    )

    state.latest_checkpoint = str(latest_path)

    return latest_path


def _count_model_parameters(
    model: torch.nn.Module,
) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_training(
    runtime,
    state: TrainingState,
) -> None:
    config = runtime.config

    torch.autograd.set_detect_anomaly(
        config.training.detect_anomaly
    )

    print("Starting vocoder training.")
    print(f"Run dir: {runtime.paths.run_dir}")
    print(f"Device: {runtime.device}")
    print(
        "Train batches per epoch: "
        f"{len(runtime.train_loader)}"
    )
    print(
        f"Physical batch size: {config.data.batch_size}"
    )
    print(
        "Gradient accumulation steps: "
        f"{config.training.gradient_accumulation_steps}"
    )
    print(
        "Effective batch size: "
        f"{config.data.batch_size * config.training.gradient_accumulation_steps}"
    )
    print(
        f"Segment frames: {config.data.segment_frames}"
    )
    print(
        "Segment samples: "
        f"{config.data.segment_frames * config.data.hop_length}"
    )

    first_epoch = state.epoch

    try:
        for epoch in range(
            state.epoch,
            config.training.epochs,
        ):
            is_first_epoch_in_process = epoch == first_epoch

            resume_seed_offset = state.global_step if (
                    runtime.resumed_from_checkpoint and is_first_epoch_in_process) else 0

            run_epoch(
                epoch=epoch,
                runtime=runtime,
                state=state,
                resume_seed_offset=resume_seed_offset,
            )

            step_vocoder_schedulers(
                components=runtime.components,
                generator_config=config.scheduler_generator,
                discriminator_config=config.scheduler_discriminator,
                interval="epoch"
            )

            state.epoch = epoch + 1
            save_latest(runtime, state)

            if runtime.stop_controller.stop_requested:
                break

    except KeyboardInterrupt:
        if state.accumulation_in_progress:
            runtime.components.optimizers.generator.zero_grad(
                set_to_none=True
            )
            runtime.components.optimizers.discriminator.zero_grad(
                set_to_none=True
            )

            print(
                "\nDiscarded incomplete accumulated gradients."
            )

        stop_path = (
            runtime.paths.checkpoint_dir
            / f"stop_step_{state.global_step:09d}.pt"
        )

        save_checkpoint(
            stop_path,
            state=state,
            components=runtime.components,
            config=config,
        )

        save_latest(runtime, state)

        print(
            f"\nTraining stopped cleanly. Saved: {stop_path}"
        )

