from __future__ import annotations

import math
from dataclasses import dataclass, field

AVERAGED_METRIC_PREFIXES = (
    "loss_",
    "discriminator_health/",
    "generator_output/",
    "grad_",
    "performance/",
)
LATEST_METRIC_PREFIXES = (
    "learning_rate_",
    "context/",
    "scheduler/",
    "amp/",
)
CRITICAL_MAXIMUM_METRICS = (
    "loss_g_total",
    "loss_d_total",
    "loss_g_mrstft_spectral_convergence",
    "grad_norm_generator",
    "grad_norm_discriminator",
    "generator_output/peak",
)


def should_average_metric(
        name: str,
) -> bool:
    return name.startswith(AVERAGED_METRIC_PREFIXES)


def split_logging_metrics(
        metrics: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    averaged: dict[str, float] = {}
    latest: dict[str, float] = {}

    for name, value in metrics.items():
        if should_average_metric(name):
            averaged[name] = float(value)
        else:
            latest[name] = float(value)

    return averaged, latest


def log_interval_maxima(
        *,
        writer,
        accumulator: ScalarMetricAccumulator,
        global_step: int,
) -> None:
    maxima = accumulator.maximums()

    for metric_name in CRITICAL_MAXIMUM_METRICS:
        if metric_name not in maxima:
            continue

        writer.add_scalar(
            f"{metric_name}_interval_maxima",
            maxima[metric_name],
            global_step,
        )


@dataclass
class ScalarMetricAccumulator:
    sums: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    maxima: dict[str, float] = field(default_factory=dict)

    def update(
            self,
            metrics: dict[str, float],
    ) -> None:
        for name, raw_value in metrics.items():
            value = float(raw_value)

            if not math.isfinite(value):
                raise RuntimeError(
                    f"Metric {name!r} is a non-finite value: {value} "
                )

            self.sums[name] = self.sums.get(name, 0) + value
            self.counts[name] = self.counts.get(name, 0) + 1
            self.maxima[name] = max(self.maxima.get(name, value), value)

    def means(self) -> dict[str, float]:
        return {
            name: total / self.counts[name]
            for name, total in self.sums.items()
            if self.counts[name] > 0
        }

    def maximums(self) -> dict[str, float]:
        return dict(self.maxima)

    def maximum_count(self) -> int:
        return max(self.counts.values(), default=0)

    def reset(self) -> None:
        self.sums.clear()
        self.counts.clear()
        self.maxima.clear()
