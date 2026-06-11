"""Tests for numerical validation and tolerance logic."""

from __future__ import annotations

import pytest
import torch

from conftest import make_artifact, make_record
from mlrw.validate import (
    attach_validation,
    check_tolerance,
    compare_outputs,
    render_markdown,
    validate_run,
)


def test_compare_identical_outputs():
    a = torch.randn(4, 10)
    metrics = compare_outputs(a, a.clone())
    assert metrics["max_abs_diff"] == 0.0
    assert metrics["mean_abs_diff"] == 0.0
    assert metrics["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["top_k_agreement"] == pytest.approx(1.0)


def test_compare_perturbed_outputs():
    torch.manual_seed(0)
    a = torch.randn(8, 20)
    b = a + 0.01 * torch.randn(8, 20)
    metrics = compare_outputs(a, b)
    assert metrics["max_abs_diff"] > 0.0
    assert 0.0 < metrics["cosine_similarity"] <= 1.0
    assert 0.0 <= metrics["top_k_agreement"] <= 1.0


def test_compare_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        compare_outputs(torch.randn(2, 3), torch.randn(2, 4))


def test_top_k_agreement_detects_reordering():
    # A monotonic vector and its reverse share no top-k indices.
    base = torch.arange(10.0).reshape(1, 10)
    reversed_ = torch.flip(base, dims=[1])
    metrics = compare_outputs(base, reversed_)
    assert metrics["top_k_agreement"] == 0.0


def test_check_tolerance_pass_and_fail():
    good = {
        "max_abs_diff": 1e-4,
        "mean_abs_diff": 1e-5,
        "cosine_similarity": 0.99999,
        "top_k_agreement": 1.0,
    }
    passed, reasons = check_tolerance("fp32", good)
    assert passed and reasons == []

    bad = {
        "max_abs_diff": 1.0,
        "mean_abs_diff": 0.1,
        "cosine_similarity": 0.5,
        "top_k_agreement": 0.1,
    }
    passed, reasons = check_tolerance("fp32", bad)
    assert not passed
    assert len(reasons) == 3


def test_bf16_tolerance_ignores_abs_diff():
    # bf16 does not gate on absolute difference, only cosine and top-k.
    metrics = {
        "max_abs_diff": 100.0,
        "mean_abs_diff": 10.0,
        "cosine_similarity": 0.999,
        "top_k_agreement": 0.99,
    }
    passed, reasons = check_tolerance("bf16", metrics)
    assert passed and reasons == []


def test_check_tolerance_unknown_precision():
    with pytest.raises(KeyError):
        check_tolerance("int8", {"max_abs_diff": 0, "cosine_similarity": 1, "top_k_agreement": 1})


def test_attach_validation_skips_baseline_and_computes_rest():
    records = [
        make_record(model="resnet18", config="eager_fp32", precision="fp32", is_baseline=True),
        make_record(model="resnet18", config="eager_bf16", precision="bf16"),
    ]
    base = torch.randn(4, 10)
    logits = {"eager_fp32": base, "eager_bf16": base + 0.001}
    attach_validation(records, logits)
    assert records[0].validation is None
    assert records[1].validation is not None
    assert records[1].validation["cosine_similarity"] > 0.99


def test_validate_run_and_markdown():
    artifact = make_artifact(
        [
            make_record(config="eager_fp32", precision="fp32", is_baseline=True),
            make_record(
                config="compile_fp32",
                precision="fp32",
                mode="compile",
                validation={
                    "max_abs_diff": 1.0,
                    "mean_abs_diff": 0.1,
                    "cosine_similarity": 0.5,
                    "top_k_agreement": 0.1,
                },
            ),
        ]
    )
    report = validate_run(artifact)
    assert not report.passed
    markdown = render_markdown(report)
    assert "TOLERANCE VIOLATION" in markdown
    assert "compile_fp32" in markdown
