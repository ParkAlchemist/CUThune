from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from src.training.vocoder.builders import (
    build_training_components,
)
from src.training.vocoder.data import (
    build_dataloaders,
    collect_overfit_batches,
    save_overfit_debug_data,
)
from src.training.vocoder.context import (
    TrainingRuntime,
)
from src.training.vocoder.runtime import (
    build_run_paths,
    create_run_directories,
    configure_stop_controller,
)
from src.training.vocoder.logging import (
    write_model_manifest,
    write_model_structure,
    write_torchinfo_generator_summary, )
from src.training.vocoder.metrics import ScalarMetricAccumulator


def build_training_runtime(
    *,
    config,
    config_path: Path,
    device: torch.device,
):
    paths = build_run_paths(config)
    create_run_directories(paths)

    train_loader, preview_loader = build_dataloaders(
        config,
        device,
    )

    overfit_batches = collect_overfit_batches(
        train_loader,
        config.training.overfit_batches,
    )

    save_overfit_debug_data(
        overfit_batches,
        paths.debug_dir,
    )

    components = build_training_components(
        config,
        device,
    )

    write_model_structure(
        generator=components.models.generator,
        discriminator=(
            components.models.discriminator
        ),
        output_path=(
                paths.run_dir / "model_structure.txt"
        ),
    )

    manifest = write_model_manifest(
        generator=components.models.generator,
        discriminator=(
            components.models.discriminator
        ),
        output_path=(
                paths.run_dir / "model_structure.json"
        ),
    )

    torchinfo_text = (
        write_torchinfo_generator_summary(
            generator=components.models.generator,
            output_path=(
                    paths.run_dir
                    / "torchinfo_generator.txt"
            ),
            cqt_bins=config.data.cqt_bins,
            segment_frames=(
                config.data.segment_frames
            ),
            device=device,
        )
    )

    writer = None

    if config.logging.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(
                log_dir=str(paths.tensorboard_dir)
            )
        except Exception as exc:
            print(
                "WARNING: TensorBoard writer unavailable: "
                f"{exc}"
            )

    if writer is not None:
        writer.add_text(
            "model/torchinfo_generator",
            f"```\n{torchinfo_text}\n```",
            global_step=0,
        )

        writer.add_scalar(
            "model/generator_parameter_count",
            manifest["generator"][
                "parameter_count"
            ],
            global_step=0,
        )

        writer.add_scalar(
            "model/discriminator_parameter_count",
            manifest["discriminator"][
                "parameter_count"
            ],
            global_step=0,
        )

    stop_controller = configure_stop_controller(
        config.control
    )

    metric_accumulator = ScalarMetricAccumulator()

    return TrainingRuntime(
        config=config,
        config_path=config_path,
        device=device,
        paths=paths,
        components=components,
        train_loader=train_loader,
        preview_loader=preview_loader,
        overfit_batches=overfit_batches,
        writer=writer,
        stop_controller=stop_controller,
        metric_accumulator=metric_accumulator
    )

