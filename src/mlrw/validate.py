"""Numerical correctness validation.

Each non-baseline configuration is compared against the fp32 eager baseline for the
same model. The comparison reduces two output tensors to four interpretable metrics
and applies per-precision tolerances. Reduced precision is expected to diverge; the
purpose of validation is to quantify that divergence and to fail when it exceeds what
the precision should produce.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .artifacts import ResultRecord, RunArtifact
from .reporting import markdown_table

TOP_K = 5


@dataclass(frozen=True)
class Tolerance:
    """Per-precision acceptance thresholds.

    A threshold set to ``None`` is not checked. Absolute difference is an upper
    bound; cosine similarity and top-k agreement are lower bounds.
    """

    max_abs_diff: float | None
    min_cosine: float
    min_top_k_agreement: float


# Defaults: fp32 compiled output should be nearly identical to fp32 eager, so the
# thresholds are tight. bf16 trades precision for speed, so the absolute-difference
# gate is relaxed and correctness is judged mainly by direction (cosine) and ranking
# (top-k agreement).
TOLERANCES: dict[str, Tolerance] = {
    "fp32": Tolerance(max_abs_diff=1.0e-2, min_cosine=0.9999, min_top_k_agreement=0.99),
    "bf16": Tolerance(max_abs_diff=None, min_cosine=0.99, min_top_k_agreement=0.90),
}


def compare_outputs(baseline: torch.Tensor, other: torch.Tensor) -> dict[str, float]:
    """Compare two output tensors and return four scalar metrics.

    The tensors are reshaped to ``(rows, features)`` where the final dimension is the
    feature axis. Cosine similarity and top-k agreement are computed per row and
    averaged, which keeps the metrics meaningful for both classification logits and
    sequence hidden states.
    """
    if baseline.shape != other.shape:
        raise ValueError(
            f"Shape mismatch: baseline {tuple(baseline.shape)} vs {tuple(other.shape)}"
        )

    base = baseline.detach().float()
    cand = other.detach().float()

    diff = (base - cand).abs()
    max_abs_diff = float(diff.max())
    mean_abs_diff = float(diff.mean())

    base2d = base.reshape(-1, base.shape[-1])
    cand2d = cand.reshape(-1, cand.shape[-1])

    cosine = torch.nn.functional.cosine_similarity(base2d, cand2d, dim=1)
    cosine_similarity = float(cosine.mean())

    k = min(TOP_K, base2d.shape[-1])
    base_topk = base2d.topk(k, dim=1).indices
    cand_topk = cand2d.topk(k, dim=1).indices
    agreement = _top_k_agreement(base_topk, cand_topk)

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "cosine_similarity": cosine_similarity,
        "top_k_agreement": agreement,
    }


def _top_k_agreement(base_topk: torch.Tensor, cand_topk: torch.Tensor) -> float:
    """Mean fraction of overlapping top-k indices per row."""
    rows = base_topk.shape[0]
    k = base_topk.shape[1]
    overlaps = 0
    for i in range(rows):
        base_set = set(base_topk[i].tolist())
        cand_set = set(cand_topk[i].tolist())
        overlaps += len(base_set & cand_set)
    return overlaps / (rows * k)


def check_tolerance(precision: str, metrics: dict[str, float]) -> tuple[bool, list[str]]:
    """Apply the tolerance for a precision; return pass flag and failure reasons."""
    tolerance = TOLERANCES.get(precision)
    if tolerance is None:
        raise KeyError(f"No tolerance defined for precision {precision!r}")

    reasons: list[str] = []
    if tolerance.max_abs_diff is not None and metrics["max_abs_diff"] > tolerance.max_abs_diff:
        reasons.append(f"max_abs_diff {metrics['max_abs_diff']:.3e} > {tolerance.max_abs_diff:.3e}")
    if metrics["cosine_similarity"] < tolerance.min_cosine:
        reasons.append(f"cosine {metrics['cosine_similarity']:.5f} < {tolerance.min_cosine:.5f}")
    if metrics["top_k_agreement"] < tolerance.min_top_k_agreement:
        agree = metrics["top_k_agreement"]
        reasons.append(f"top_k_agreement {agree:.3f} < {tolerance.min_top_k_agreement:.3f}")
    return (not reasons), reasons


@dataclass
class ConfigValidation:
    """Validation outcome for a single configuration."""

    model: str
    config: str
    precision: str
    metrics: dict[str, float]
    passed: bool
    reasons: list[str]


@dataclass
class ValidationReport:
    """Aggregate validation outcome over an artifact."""

    entries: list[ConfigValidation]

    @property
    def passed(self) -> bool:
        return all(entry.passed for entry in self.entries)


def validate_run(artifact: RunArtifact) -> ValidationReport:
    """Validate the stored comparison metrics of every non-baseline configuration."""
    entries: list[ConfigValidation] = []
    for record in artifact.results:
        if record.is_baseline or record.validation is None:
            continue
        passed, reasons = check_tolerance(record.precision, record.validation)
        entries.append(
            ConfigValidation(
                model=record.model,
                config=record.config,
                precision=record.precision,
                metrics=record.validation,
                passed=passed,
                reasons=reasons,
            )
        )
    return ValidationReport(entries=entries)


def attach_validation(records: list[ResultRecord], logits: dict[str, torch.Tensor]) -> None:
    """Populate the ``validation`` field of each non-baseline record in place.

    ``logits`` maps a record's configuration name to its output tensor. The baseline
    record for each model supplies the reference tensor.
    """
    baseline_logits: dict[str, torch.Tensor] = {
        record.model: logits[record.config] for record in records if record.is_baseline
    }
    for record in records:
        if record.is_baseline:
            continue
        baseline = baseline_logits[record.model]
        record.validation = compare_outputs(baseline, logits[record.config])


def render_markdown(report: ValidationReport) -> str:
    """Render a human-readable validation report in Markdown."""
    lines = ["# Numerical Validation Report", ""]
    if not report.entries:
        lines.append("No non-baseline configurations were validated.")
        return "\n".join(lines) + "\n"

    status = "PASS" if report.passed else "TOLERANCE VIOLATION"
    lines.append(f"Overall status: **{status}**")
    lines.append("")
    lines.append("All non-baseline configurations are compared against the fp32 eager baseline.")
    lines.append("")

    header = ["Model", "Config", "Max abs diff", "Mean abs diff", "Cosine", "Top-5 agree", "Status"]
    rows = []
    for entry in report.entries:
        m = entry.metrics
        rows.append(
            [
                entry.model,
                entry.config,
                f"{m['max_abs_diff']:.3e}",
                f"{m['mean_abs_diff']:.3e}",
                f"{m['cosine_similarity']:.5f}",
                f"{m['top_k_agreement']:.3f}",
                "PASS" if entry.passed else "FAIL",
            ]
        )
    lines.append(markdown_table(header, rows))

    failures = [entry for entry in report.entries if not entry.passed]
    if failures:
        lines.append("")
        lines.append("## Tolerance violations")
        lines.append("")
        for entry in failures:
            reasons = "; ".join(entry.reasons)
            lines.append(f"- {entry.model} / {entry.config}: {reasons}")
    return "\n".join(lines) + "\n"
