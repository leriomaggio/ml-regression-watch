"""Latency comparison and divergence-by-depth plots.

A grouped bar chart summarises median latency per configuration for each model, and a
line chart traces relative error against depth for one or more configurations. The
plots are written as PNGs and are used both in the README and as CI artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .artifacts import RunArtifact  # noqa: E402

if TYPE_CHECKING:
    from .localize import LocalizationReport


def plot_latency_comparison(artifact: RunArtifact, path: str | Path) -> Path:
    """Render median latency per configuration, grouped by model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    models = artifact.models
    configs = _ordered_configs(artifact)
    latency = {(r.model, r.config): r.metrics.median_latency_ms for r in artifact.results}

    fig, ax = plt.subplots(figsize=(9, 5))
    n_configs = len(configs)
    group_width = 0.8
    bar_width = group_width / max(n_configs, 1)

    for i, config in enumerate(configs):
        offsets = [j + (i - (n_configs - 1) / 2) * bar_width for j in range(len(models))]
        heights = [latency.get((model, config), 0.0) for model in models]
        ax.bar(offsets, heights, width=bar_width, label=config)

    # Latency across configurations spans more than an order of magnitude, so a
    # logarithmic axis keeps every bar legible rather than letting the slowest
    # configuration flatten the rest.
    ax.set_yscale("log")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel("Median latency (ms, log scale)")
    ax.set_title("Median latency by model and execution configuration")
    ax.legend(title="Configuration")
    ax.grid(axis="y", linestyle=":", alpha=0.5, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_divergence_by_depth(
    reports: list[LocalizationReport], path: str | Path
) -> Path:
    """Render relative error against capture-point depth, one line per configuration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for report in reports:
        points = [layer.point for layer in report.layers]
        # Relative error spans several orders of magnitude and a log axis cannot show a
        # genuine zero, so the floor keeps fp32-noise points on the chart.
        values = [max(layer.relative_error, 1e-12) for layer in report.layers]
        ax.plot(range(len(points)), values, marker="o", label=report.config)

    if reports:
        labels = [layer.point for layer in reports[0].layers]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_yscale("log")
    ax.set_ylabel("Relative error (log scale)")
    ax.set_xlabel("Capture point (depth)")
    ax.set_title("Divergence from the fp32 eager baseline by depth")
    ax.legend(title="Configuration")
    ax.grid(axis="y", linestyle=":", alpha=0.5, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _ordered_configs(artifact: RunArtifact) -> list[str]:
    """Return distinct configuration names in artifact order."""
    seen: list[str] = []
    for record in artifact.results:
        if record.config not in seen:
            seen.append(record.config)
    return seen
