"""Tests for statistical regression detection."""

from __future__ import annotations

from conftest import make_artifact, make_record
from mlrw.compare import compare_samples, detect_regressions, render_markdown


def _stable(center: float, n: int = 12) -> list[float]:
    """A small deterministic sample clustered around a center value."""
    offsets = [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    return [center + o for o in offsets[:n]]


def test_no_regression_for_similar_samples():
    baseline = _stable(10.0)
    current = _stable(10.0)
    delta, p_value, effect, is_regression = compare_samples(current, baseline)
    assert not is_regression
    assert abs(delta) < 0.05


def test_injected_slowdown_is_flagged():
    baseline = _stable(10.0)
    current = _stable(13.0)  # 30 percent slower
    delta, p_value, effect, is_regression = compare_samples(current, baseline)
    assert is_regression
    assert delta > 0.05
    assert p_value < 0.05
    assert effect > 0.0


def test_small_consistent_shift_below_margin_not_flagged():
    # A 3 percent shift is statistically detectable but below the 5 percent margin.
    baseline = _stable(10.0)
    current = _stable(10.3)
    delta, p_value, effect, is_regression = compare_samples(current, baseline)
    assert p_value < 0.05  # detectable
    assert not is_regression  # but below the practical margin


def test_speedup_is_not_a_regression():
    baseline = _stable(10.0)
    current = _stable(7.0)
    delta, p_value, effect, is_regression = compare_samples(current, baseline)
    assert not is_regression
    assert delta < 0.0


def test_identical_constant_samples_not_flagged():
    delta, p_value, effect, is_regression = compare_samples([5.0] * 5, [5.0] * 5)
    assert not is_regression
    assert delta == 0.0


def test_custom_margin_changes_outcome():
    baseline = _stable(10.0)
    current = _stable(10.8)  # 8 percent slower
    _, _, _, default_flag = compare_samples(current, baseline, margin=0.05)
    _, _, _, strict_flag = compare_samples(current, baseline, margin=0.20)
    assert default_flag  # exceeds 5 percent
    assert not strict_flag  # within 20 percent


def test_detect_regressions_over_artifacts():
    baseline = make_artifact(
        [
            make_record(config="eager_fp32", is_baseline=True, samples=_stable(10.0)),
            make_record(config="eager_bf16", precision="bf16", samples=_stable(6.0)),
        ]
    )
    current = make_artifact(
        [
            make_record(config="eager_fp32", is_baseline=True, samples=_stable(10.0)),
            make_record(config="eager_bf16", precision="bf16", samples=_stable(9.0)),
        ]
    )
    report = detect_regressions(current, baseline)
    assert report.has_regression
    flagged = [c for c in report.comparisons if c.is_regression]
    assert len(flagged) == 1
    assert flagged[0].config == "eager_bf16"

    markdown = render_markdown(report)
    assert "REGRESSION" in markdown
    assert "eager_bf16" in markdown
