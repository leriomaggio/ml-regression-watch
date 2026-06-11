"""Tests for the execution configuration model."""

from __future__ import annotations

import pytest

from mlrw.config import (
    Device,
    ExecConfig,
    Mode,
    Precision,
    default_configs,
    resolve_device,
)


def test_exec_config_name_and_baseline():
    baseline = ExecConfig(Device.CPU, Precision.FP32, Mode.EAGER)
    assert baseline.name == "eager_fp32"
    assert baseline.is_baseline

    other = ExecConfig(Device.CPU, Precision.BF16, Mode.COMPILE)
    assert other.name == "compile_bf16"
    assert not other.is_baseline


def test_default_configs_matrix_size_and_baseline_first():
    configs = default_configs(Device.CPU)
    assert len(configs) == 4
    names = {c.name for c in configs}
    assert names == {"eager_fp32", "eager_bf16", "compile_fp32", "compile_bf16"}
    # The baseline must be first so downstream code can rely on its position.
    assert configs[0].is_baseline


def test_adding_a_precision_extends_matrix():
    # Adding a precision axis is a single-argument change, demonstrating extensibility.
    configs = default_configs(
        Device.CPU,
        precisions=(Precision.FP32, Precision.BF16),
        modes=(Mode.EAGER,),
    )
    assert len(configs) == 2
    assert all(c.mode is Mode.EAGER for c in configs)


def test_resolve_device_explicit():
    assert resolve_device("cpu") is Device.CPU
    assert resolve_device("CPU") is Device.CPU


def test_resolve_device_auto(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") is Device.CPU
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") is Device.CUDA


def test_resolve_device_invalid():
    with pytest.raises(ValueError):
        resolve_device("tpu")
