"""Tests for per-layer divergence localisation.

The tests exercise the capture mechanism on a tiny synthetic two-block module rather
than downloading DistilBERT, and check the metric, first-divergence, and configuration
parsing logic on hand-constructed tensors.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from typer.testing import CliRunner

import mlrw.cli as cli_module
from mlrw.config import Device, ExecConfig, Mode, Precision
from mlrw.localize import (
    CAPTURE_POINTS,
    LayerDivergence,
    capture_activations,
    compare_activations,
    first_divergence,
    register_capture_hooks,
)

runner = CliRunner()


class _Block(nn.Module):
    """A transformer-like block that returns its hidden state in a length-one tuple."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (torch.relu(self.linear(x)),)


class _TwoBlock(nn.Module):
    """A minimal two-block network mirroring the embedding and block boundaries."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.embeddings = nn.Linear(dim, dim)
        self.block_0 = _Block(dim)
        self.block_1 = _Block(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.embeddings(x)
        hidden = self.block_0(hidden)[0]
        hidden = self.block_1(hidden)[0]
        return hidden


def _two_block_targets(model: _TwoBlock):
    return [
        ("embeddings", model.embeddings),
        ("block_0", model.block_0),
        ("block_1", model.block_1),
    ]


def test_hooks_capture_every_point():
    torch.manual_seed(0)
    model = _TwoBlock().eval()
    store: dict[str, torch.Tensor] = {}
    handles = register_capture_hooks(_two_block_targets(model), store)
    with torch.inference_mode():
        model(torch.randn(4, 8))
    for handle in handles:
        handle.remove()

    assert set(store) == {"embeddings", "block_0", "block_1"}
    assert all(tensor.shape == (4, 8) for tensor in store.values())
    # Activations are upcast to fp32 on CPU inside the hook.
    assert all(tensor.dtype == torch.float32 for tensor in store.values())
    assert all(tensor.device.type == "cpu" for tensor in store.values())


def test_capture_activations_reads_final_hidden_state():
    torch.manual_seed(0)
    model = _TwoBlock().eval()
    inputs = {"x": torch.randn(4, 8)}
    config = ExecConfig(Device.CPU, Precision.FP32, Mode.EAGER)
    store = capture_activations(model, inputs, config, targets=_two_block_targets(model))

    assert "last_hidden_state" in store
    # The final block boundary and the model output carry the same tensor.
    assert torch.equal(store["block_1"], store["last_hidden_state"])


def test_compare_identical_activations_is_zero():
    base = {point: torch.randn(4, 8) for point in CAPTURE_POINTS}
    target = {point: tensor.clone() for point, tensor in base.items()}
    layers = compare_activations(base, target)

    assert [layer.point for layer in layers] == list(CAPTURE_POINTS)
    for layer in layers:
        assert layer.max_abs_diff == 0.0
        assert layer.relative_error == 0.0
        assert layer.cosine_similarity == pytest.approx(1.0, abs=1e-6)


def test_compare_metrics_on_constructed_tensors():
    base = {"embeddings": torch.tensor([[3.0, 4.0]])}  # norm 5
    target = {"embeddings": torch.tensor([[3.0, 0.0]])}  # differs by 4 in one element
    (layer,) = compare_activations(base, target)

    assert layer.max_abs_diff == pytest.approx(4.0)
    assert layer.mean_abs_diff == pytest.approx(2.0)
    assert layer.relative_error == pytest.approx(4.0 / 5.0)


def test_compare_shape_mismatch_raises():
    base = {"embeddings": torch.randn(2, 3)}
    target = {"embeddings": torch.randn(2, 4)}
    with pytest.raises(ValueError, match="Shape mismatch"):
        compare_activations(base, target)


def test_compare_frees_consumed_tensors():
    base = {"embeddings": torch.randn(2, 3)}
    target = {"embeddings": torch.randn(2, 3)}
    compare_activations(base, target)
    # Each capture point is popped from both sets once its metrics are computed.
    assert base == {}
    assert target == {}


def _layer(point: str, relative_error: float) -> LayerDivergence:
    return LayerDivergence(
        point=point,
        max_abs_diff=relative_error,
        mean_abs_diff=relative_error,
        cosine_similarity=1.0,
        relative_error=relative_error,
    )


def test_first_divergence_detects_injected_layer():
    layers = [
        _layer("embeddings", 1e-7),
        _layer("block_0", 1e-7),
        _layer("block_1", 5e-3),
        _layer("block_2", 1e-2),
    ]
    assert first_divergence(layers, threshold=1e-3) == "block_1"


def test_first_divergence_returns_none_below_threshold():
    layers = [_layer("embeddings", 1e-7), _layer("block_0", 1e-6)]
    assert first_divergence(layers, threshold=1e-3) is None


def test_localize_config_parsing_valid():
    config = cli_module._resolve_localize_config("eager_bf16", "cpu")
    assert config.mode is Mode.EAGER
    assert config.precision is Precision.BF16
    assert config.device is Device.CPU


def test_localize_config_parsing_rejects_baseline():
    from typer import BadParameter

    with pytest.raises(BadParameter):
        cli_module._resolve_localize_config("eager_fp32", "cpu")


def test_localize_config_parsing_rejects_malformed():
    from typer import BadParameter

    with pytest.raises(BadParameter):
        cli_module._resolve_localize_config("nonsense", "cpu")
    with pytest.raises(BadParameter):
        cli_module._resolve_localize_config("turbo_fp32", "cpu")


def test_localize_cli_runs_on_synthetic_model(tmp_path, monkeypatch):
    # Drive the command end to end without downloading DistilBERT by substituting a
    # tiny model and its inputs.
    import mlrw.localize as localize_module

    def tiny_targets(model):
        return _two_block_targets(model)

    monkeypatch.setattr(localize_module, "distilbert_capture_targets", tiny_targets)

    def fake_model_factory():
        torch.manual_seed(0)
        return _TwoBlock().eval()

    def fake_input_factory():
        torch.manual_seed(0)
        return {"x": torch.randn(4, 8)}

    import mlrw.models as models_module

    monkeypatch.setattr(models_module, "load_distilbert", fake_model_factory)
    monkeypatch.setattr(models_module, "make_distilbert_input", fake_input_factory)

    report = tmp_path / "localize.md"
    plot = tmp_path / "localize.png"
    result = runner.invoke(
        cli_module.app,
        ["localize", "eager_bf16", "--device", "cpu", "--report", str(report), "--plot", str(plot)],
    )
    assert result.exit_code == 0, result.stdout
    assert "Divergence Localisation Report" in report.read_text(encoding="utf-8")
    assert plot.exists() and plot.stat().st_size > 0
