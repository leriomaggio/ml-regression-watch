"""Tests for the benchmark runner on a tiny model."""

from __future__ import annotations

import pytest

from conftest import make_tiny_input, make_tiny_model
from mlrw.config import Device, ExecConfig, Mode, Precision
from mlrw.runner import _median, _percentile, benchmark


def test_benchmark_produces_metrics_and_samples():
    model = make_tiny_model()
    inputs = make_tiny_input()
    config = ExecConfig(Device.CPU, Precision.FP32, Mode.EAGER)
    result = benchmark(model, inputs, config, warmup=2, iters=5, repeats=4)

    assert len(result.latency_samples_ms) == 4
    assert result.metrics.median_latency_ms >= 0.0
    assert result.metrics.p90_latency_ms >= result.metrics.median_latency_ms
    assert result.metrics.throughput_items_per_s > 0.0
    assert result.logits.shape == (4, 8)
    assert not result.compile_failed


def test_bf16_autocast_runs_and_matches_shape():
    model = make_tiny_model()
    inputs = make_tiny_input()
    config = ExecConfig(Device.CPU, Precision.BF16, Mode.EAGER)
    result = benchmark(model, inputs, config, warmup=1, iters=3, repeats=2)
    assert result.logits.shape == (4, 8)


def test_median_helper():
    assert _median([3.0, 1.0, 2.0]) == 2.0
    assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert _median([]) == 0.0


def test_percentile_helper():
    values = [float(i) for i in range(1, 11)]  # 1..10
    assert _percentile(values, 0) == 1.0
    assert _percentile(values, 100) == 10.0
    assert _percentile(values, 50) == pytest.approx(5.5)
    assert _percentile([], 90) == 0.0
