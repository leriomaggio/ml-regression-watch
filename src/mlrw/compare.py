"""Performance regression detection.

Regression detection compares a current run against a stored baseline. Rather than a
naive threshold on a single latency number, it treats the per-repeat latency samples
of each configuration as a distribution and applies the Mann-Whitney U test. A
configuration is flagged as a regression only when the current latencies are both
statistically greater than the baseline (one-sided test) and larger by more than a
configurable relative margin. Requiring both conditions guards against flagging
differences that are statistically detectable but practically negligible.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import mannwhitneyu

from .artifacts import RunArtifact
from .reporting import markdown_table

DEFAULT_MARGIN = 0.05
DEFAULT_ALPHA = 0.05


@dataclass
class MetricComparison:
    """Regression outcome for a single (model, configuration)."""

    model: str
    config: str
    baseline_median_ms: float
    current_median_ms: float
    relative_delta: float
    p_value: float
    effect_size: float
    is_regression: bool

    @property
    def status(self) -> str:
        return "REGRESSION" if self.is_regression else "PASS"


@dataclass
class CompareReport:
    """Aggregate regression outcome of a comparison."""

    comparisons: list[MetricComparison]
    baseline_present: bool = True

    @property
    def has_regression(self) -> bool:
        return any(c.is_regression for c in self.comparisons)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _rank_biserial(current: list[float], baseline: list[float], u_statistic: float) -> float:
    """Rank-biserial correlation as a bounded effect size in ``[-1, 1]``.

    A positive value indicates that current latencies tend to exceed baseline
    latencies. The statistic is derived from the same U used by the test.
    """
    n1 = len(current)
    n2 = len(baseline)
    if n1 == 0 or n2 == 0:
        return 0.0
    # u_statistic is the U for the current sample; with this orientation a value
    # approaching +1 means current latencies rank above baseline latencies.
    return (2.0 * u_statistic) / (n1 * n2) - 1.0


def compare_samples(
    current: list[float],
    baseline: list[float],
    *,
    margin: float = DEFAULT_MARGIN,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float, float, bool]:
    """Compare two latency samples.

    Returns the relative median delta, the p-value of the one-sided Mann-Whitney U
    test, the rank-biserial effect size, and whether the pair is a regression. The
    test direction asks whether current latencies are stochastically greater than the
    baseline latencies.
    """
    base_median = _median(baseline)
    cur_median = _median(current)
    relative_delta = (cur_median - base_median) / base_median if base_median > 0 else 0.0

    # The U test requires variation in at least one sample; identical constant
    # samples yield no evidence of a difference.
    if len(set(current)) == 1 and len(set(baseline)) == 1 and current[0] == baseline[0]:
        return relative_delta, 1.0, 0.0, False

    result = mannwhitneyu(current, baseline, alternative="greater")
    p_value = float(result.pvalue)
    effect_size = _rank_biserial(current, baseline, float(result.statistic))
    is_regression = p_value < alpha and relative_delta > margin
    return relative_delta, p_value, effect_size, is_regression


def detect_regressions(
    current: RunArtifact,
    baseline: RunArtifact,
    *,
    margin: float = DEFAULT_MARGIN,
    alpha: float = DEFAULT_ALPHA,
) -> CompareReport:
    """Detect latency regressions of a current run against a baseline run."""
    baseline_index = {(r.model, r.config): r for r in baseline.results}
    comparisons: list[MetricComparison] = []
    for record in current.results:
        key = (record.model, record.config)
        base_record = baseline_index.get(key)
        if base_record is None:
            continue
        relative_delta, p_value, effect_size, is_regression = compare_samples(
            record.latency_samples_ms,
            base_record.latency_samples_ms,
            margin=margin,
            alpha=alpha,
        )
        comparisons.append(
            MetricComparison(
                model=record.model,
                config=record.config,
                baseline_median_ms=_median(base_record.latency_samples_ms),
                current_median_ms=_median(record.latency_samples_ms),
                relative_delta=relative_delta,
                p_value=p_value,
                effect_size=effect_size,
                is_regression=is_regression,
            )
        )
    return CompareReport(comparisons=comparisons, baseline_present=True)


def render_markdown(report: CompareReport, *, margin: float = DEFAULT_MARGIN) -> str:
    """Render a regression report in Markdown."""
    lines = ["# Performance Regression Report", ""]
    if not report.baseline_present:
        lines.append("No baseline was found. Regression detection was skipped.")
        return "\n".join(lines) + "\n"
    if not report.comparisons:
        lines.append("The baseline and current run share no comparable configurations.")
        return "\n".join(lines) + "\n"

    status = "REGRESSION" if report.has_regression else "PASS"
    lines.append(f"Overall status: **{status}**")
    lines.append("")
    lines.append(
        f"Latency samples are compared with a one-sided Mann-Whitney U test "
        f"(alpha {DEFAULT_ALPHA}) and a relative margin of {margin:.0%}."
    )
    lines.append("")

    header = [
        "Model",
        "Config",
        "Baseline ms",
        "Current ms",
        "Delta",
        "p-value",
        "Effect",
        "Status",
    ]
    rows = []
    for c in report.comparisons:
        rows.append(
            [
                c.model,
                c.config,
                f"{c.baseline_median_ms:.2f}",
                f"{c.current_median_ms:.2f}",
                f"{c.relative_delta:+.1%}",
                f"{c.p_value:.3f}",
                f"{c.effect_size:+.2f}",
                c.status,
            ]
        )
    lines.append(markdown_table(header, rows))
    return "\n".join(lines) + "\n"
