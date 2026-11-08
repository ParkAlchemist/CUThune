import pytest
import torch


from vocoder.src.nes_vocoder.config.schema import SchedulerConfig
from vocoder.src.nes_vocoder.training.builders import build_scheduler, build_fixed_floor_cosine_scheduler
from vocoder.src.nes_vocoder.training.schedulers import step_scheduler


def test_build_exponential_scheduler() -> None:
    model = torch.nn.Linear(4, 1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
    )

    config = SchedulerConfig(
        enabled=True,
        name="exponential",
        interval="epoch",
        gamma=0.9,
    )

    scheduler = build_scheduler(
        optimizer,
        config,
    )

    assert isinstance(
        scheduler,
        torch.optim.lr_scheduler.ExponentialLR,
    )


def test_disabled_scheduler_returns_none() -> None:
    model = torch.nn.Linear(4, 1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
    )

    config = SchedulerConfig(
        enabled=False,
        name="none",
    )

    assert build_scheduler(
        optimizer,
        config,
    ) is None


def test_epoch_scheduler_ignores_optimizer_interval() -> None:
    model = torch.nn.Linear(4, 1)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1.0,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.5,
    )

    config = SchedulerConfig(
        enabled=True,
        name="exponential",
        interval="epoch",
        gamma=0.5,
    )

    step_scheduler(
        scheduler,
        config,
        interval="optimizer_step",
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        1.0
    )

    optimizer.step()

    step_scheduler(
        scheduler,
        config,
        interval="epoch",
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        0.5
    )


def test_scheduler_state_roundtrip() -> None:
    first_model = torch.nn.Linear(4, 1)
    first_optimizer = torch.optim.AdamW(
        first_model.parameters(),
        lr=1e-3,
    )
    first_scheduler = (
        torch.optim.lr_scheduler.ExponentialLR(
            first_optimizer,
            gamma=0.9,
        )
    )

    for _ in range(3):
        first_optimizer.step()
        first_scheduler.step()

    saved_optimizer = first_optimizer.state_dict()
    saved_scheduler = first_scheduler.state_dict()

    second_model = torch.nn.Linear(4, 1)
    second_optimizer = torch.optim.AdamW(
        second_model.parameters(),
        lr=1e-3,
    )

    # Scheduler exists before optimizer state restoration.
    second_scheduler = (
        torch.optim.lr_scheduler.ExponentialLR(
            second_optimizer,
            gamma=0.9,
        )
    )

    second_optimizer.load_state_dict(
        saved_optimizer
    )
    second_scheduler.load_state_dict(
        saved_scheduler
    )

    assert second_scheduler.last_epoch == (
        first_scheduler.last_epoch
    )

    assert second_optimizer.param_groups[0]["lr"] == (
        pytest.approx(
            first_optimizer.param_groups[0]["lr"]
        )
    )


def test_fixed_floor_cosine_reaches_and_holds_minimum() -> None:
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1.0,
    )

    scheduler = build_fixed_floor_cosine_scheduler(
        optimizer,
        t_max=10,
        eta_min=0.1,
    )

    learning_rates = []

    for _ in range(15):
        optimizer.step()
        scheduler.step()

        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates[9] == pytest.approx(0.1, abs=1e-6)

    assert all(learning_rate == pytest.approx(0.1, abs=1e-6) for learning_rate in learning_rates[9:])


