"""Tests for the command-line interface."""

from __future__ import annotations

from typer.testing import CliRunner

import mlrw.cli as cli_module
import mlrw.models as models_module
from conftest import make_artifact, make_record, make_tiny_input, make_tiny_model
from mlrw.artifacts import load_artifact, save_artifact
from mlrw.config import Device, ExecConfig, Mode, Precision
from mlrw.models import ModelSpec

runner = CliRunner()


def _eager_configs(device: Device) -> list[ExecConfig]:
    return [
        ExecConfig(device, Precision.FP32, Mode.EAGER),
        ExecConfig(device, Precision.BF16, Mode.EAGER),
    ]


def _tiny_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            key="tiny",
            task="vision",
            loader=make_tiny_model,
            input_factory=make_tiny_input,
        )
    ]


def test_version():
    result = runner.invoke(cli_module.app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_run_writes_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_module, "default_configs", _eager_configs)
    monkeypatch.setattr(models_module, "default_model_specs", _tiny_specs)

    out = tmp_path / "run.json"
    result = runner.invoke(
        cli_module.app,
        [
            "run",
            "-o",
            str(out),
            "--device",
            "cpu",
            "--warmup",
            "1",
            "--iters",
            "2",
            "--repeats",
            "3",
        ],
    )
    assert result.exit_code == 0, result.stdout
    artifact = load_artifact(out)
    assert artifact.models == ["tiny"]
    assert len(artifact.results) == 2
    baseline = [r for r in artifact.results if r.is_baseline]
    assert len(baseline) == 1
    non_baseline = [r for r in artifact.results if not r.is_baseline]
    assert non_baseline[0].validation is not None


def test_validate_passing_artifact(tmp_path):
    artifact = make_artifact(
        [
            make_record(config="eager_fp32", precision="fp32", is_baseline=True),
            make_record(
                config="eager_bf16",
                precision="bf16",
                validation={
                    "max_abs_diff": 0.5,
                    "mean_abs_diff": 0.05,
                    "cosine_similarity": 0.999,
                    "top_k_agreement": 0.99,
                },
            ),
        ]
    )
    path = save_artifact(artifact, tmp_path / "run.json")
    result = runner.invoke(cli_module.app, ["validate", str(path)])
    assert result.exit_code == 0
    assert "PASS" in result.stdout


def test_validate_failing_artifact_exits_nonzero(tmp_path):
    artifact = make_artifact(
        [
            make_record(config="eager_fp32", precision="fp32", is_baseline=True),
            make_record(
                config="eager_bf16",
                precision="bf16",
                validation={
                    "max_abs_diff": 10.0,
                    "mean_abs_diff": 1.0,
                    "cosine_similarity": 0.1,
                    "top_k_agreement": 0.1,
                },
            ),
        ]
    )
    path = save_artifact(artifact, tmp_path / "run.json")
    result = runner.invoke(cli_module.app, ["validate", str(path)])
    assert result.exit_code == 1
    assert "TOLERANCE VIOLATION" in result.stdout


def test_compare_missing_baseline_skips(tmp_path):
    current = save_artifact(make_artifact([make_record(is_baseline=True)]), tmp_path / "run.json")
    missing = tmp_path / "no_baseline.json"
    result = runner.invoke(
        cli_module.app, ["compare", "--baseline", str(missing), "--current", str(current)]
    )
    assert result.exit_code == 0
    assert "skipping" in result.stdout.lower()


def test_compare_detects_regression_exits_nonzero(tmp_path):
    baseline = save_artifact(
        make_artifact([make_record(is_baseline=True, samples=[10.0, 10.1, 9.9, 10.0, 10.2, 9.8])]),
        tmp_path / "baseline.json",
    )
    current = save_artifact(
        make_artifact(
            [make_record(is_baseline=True, samples=[14.0, 14.1, 13.9, 14.0, 14.2, 13.8])]
        ),
        tmp_path / "run.json",
    )
    result = runner.invoke(
        cli_module.app, ["compare", "--baseline", str(baseline), "--current", str(current)]
    )
    assert result.exit_code == 1
    assert "REGRESSION" in result.stdout


def test_compare_no_fail_on_regression_exits_zero(tmp_path):
    baseline = save_artifact(
        make_artifact(
            [make_record(is_baseline=True, samples=[10.0, 10.1, 9.9, 10.0, 10.2, 9.8])]
        ),
        tmp_path / "baseline.json",
    )
    current = save_artifact(
        make_artifact(
            [make_record(is_baseline=True, samples=[14.0, 14.1, 13.9, 14.0, 14.2, 13.8])]
        ),
        tmp_path / "run.json",
    )
    result = runner.invoke(
        cli_module.app,
        [
            "compare",
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--no-fail-on-regression",
        ],
    )
    assert result.exit_code == 0
    assert "REGRESSION" in result.stdout


def test_update_baseline(tmp_path):
    source = save_artifact(make_artifact([make_record(is_baseline=True)]), tmp_path / "run.json")
    out = tmp_path / "baselines" / "cpu_ci.json"
    result = runner.invoke(
        cli_module.app, ["update-baseline", "--from", str(source), "--out", str(out)]
    )
    assert result.exit_code == 0
    assert out.exists()
    assert load_artifact(out).models == ["resnet18"]


def test_plot_writes_png(tmp_path):
    artifact = make_artifact(
        [
            make_record(model="resnet18", config="eager_fp32", is_baseline=True),
            make_record(model="resnet18", config="eager_bf16", precision="bf16"),
        ]
    )
    path = save_artifact(artifact, tmp_path / "run.json")
    out = tmp_path / "plot.png"
    result = runner.invoke(cli_module.app, ["plot", str(path), "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert out.stat().st_size > 0
