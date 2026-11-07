import pytest

from vocoder.src.nes_vocoder.training.metrics import should_average_metric, ScalarMetricAccumulator


def test_metric_accumulator_calculates_means() -> None:
    accumulator = ScalarMetricAccumulator()

    accumulator.update(
        {
            "loss_g_total": 1.0,
            "loss_d_total": 4.0,
        }
    )

    accumulator.update(
        {
            "loss_g_total": 3.0,
            "loss_d_total": 8.0,
        }
    )

    means = accumulator.means()

    assert means[
        "loss_g_total"
    ] == pytest.approx(2.0)

    assert means[
        "loss_d_total"
    ] == pytest.approx(6.0)


def test_metric_accumulator_tracks_individual_counts() -> None:
    accumulator = ScalarMetricAccumulator()

    accumulator.update(
        {
            "loss_g_total": 1.0,
        }
    )

    accumulator.update(
        {
            "loss_g_total": 3.0,
            "optional_metric": 10.0,
        }
    )

    means = accumulator.means()

    assert means[
        "loss_g_total"
    ] == pytest.approx(2.0)

    assert means[
        "optional_metric"
    ] == pytest.approx(10.0)

    assert accumulator.counts[
        "loss_g_total"
    ] == 2

    assert accumulator.counts[
        "optional_metric"
    ] == 1


def test_metric_accumulator_tracks_maxima() -> None:
    accumulator = ScalarMetricAccumulator()

    for value in (1.0, 8.0, 3.0):
        accumulator.update(
            {
                "generator_grad_norm": value,
            }
        )

    assert accumulator.maximums()[
        "generator_grad_norm"
    ] == pytest.approx(8.0)


def test_metric_accumulator_reset() -> None:
    accumulator = ScalarMetricAccumulator()

    accumulator.update(
        {
            "loss_g_total": 2.0,
        }
    )

    accumulator.reset()

    assert accumulator.sums == {}
    assert accumulator.counts == {}
    assert accumulator.maxima == {}
    assert accumulator.means() == {}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("loss_g_total", True),
        (
            "discriminator_health/mpd/real_score_mean",
            True,
        ),
        (
            "generator_output/rms",
            True,
        ),
        (
            "grad_norm_generator",
            True,
        ),
        (
            "learning_rate_generator",
            False,
        ),
        (
            "context/convnext_00/layer_scale_mean",
            False,
        ),
    ],
)
def test_metric_averaging_classification(
    name: str,
    expected: bool,
) -> None:
    assert (
        should_average_metric(name)
        is expected
    )


