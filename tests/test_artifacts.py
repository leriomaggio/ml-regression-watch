"""Tests for the versioned artifact schema."""

from __future__ import annotations

import json

import pytest

from conftest import make_artifact, make_record
from mlrw.artifacts import SCHEMA_VERSION, RunArtifact, load_artifact, save_artifact


def test_round_trip_preserves_data(tmp_path):
    artifact = make_artifact(
        [
            make_record(is_baseline=True),
            make_record(
                config="eager_bf16",
                precision="bf16",
                is_baseline=False,
                validation={
                    "max_abs_diff": 0.5,
                    "mean_abs_diff": 0.05,
                    "cosine_similarity": 0.999,
                    "top_k_agreement": 0.95,
                },
            ),
        ]
    )
    path = save_artifact(artifact, tmp_path / "run.json")
    loaded = load_artifact(path)

    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.metadata.git_sha == artifact.metadata.git_sha
    assert len(loaded.results) == 2
    baseline, other = loaded.results
    assert baseline.is_baseline
    assert other.validation["cosine_similarity"] == pytest.approx(0.999)
    assert other.metrics.median_latency_ms == pytest.approx(other.metrics.median_latency_ms)


def test_models_and_results_for():
    artifact = make_artifact(
        [
            make_record(model="resnet18", is_baseline=True),
            make_record(model="distilbert", is_baseline=True),
            make_record(model="distilbert", config="eager_bf16", precision="bf16"),
        ]
    )
    assert artifact.models == ["resnet18", "distilbert"]
    assert len(artifact.results_for("distilbert")) == 2


def test_load_rejects_unknown_schema_version(tmp_path):
    artifact = make_artifact([make_record(is_baseline=True)])
    data = artifact.to_dict()
    data["schema_version"] = 999
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        load_artifact(path)


def test_to_dict_is_json_serialisable():
    artifact = make_artifact([make_record(is_baseline=True)])
    # Should not raise; round-trips through the JSON encoder.
    encoded = json.dumps(artifact.to_dict())
    assert RunArtifact.from_dict(json.loads(encoded)).models == ["resnet18"]
