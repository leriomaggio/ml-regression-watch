"""Versioned JSON artifacts.

A run artifact is a self-describing record of a benchmark run. It contains the
schema version, run metadata (timestamp, git SHA, hardware, library versions), and
one result record per (model, configuration) pair. Artifacts are the unit of
exchange between the ``run``, ``validate``, and ``compare`` commands, so the schema
is versioned and validated on load.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class Metrics:
    """Performance metrics for a single configuration."""

    median_latency_ms: float
    p90_latency_ms: float
    throughput_items_per_s: float
    peak_memory_mb: float


@dataclass
class ResultRecord:
    """A single (model, configuration) result.

    ``latency_samples_ms`` holds one latency measurement per repeat and is the
    sample that the regression test consumes. ``validation`` is populated for every
    non-baseline configuration and is ``None`` for the baseline itself.
    """

    model: str
    config: str
    device: str
    precision: str
    mode: str
    is_baseline: bool
    metrics: Metrics
    latency_samples_ms: list[float]
    validation: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResultRecord:
        metrics = Metrics(**data["metrics"])
        return cls(
            model=data["model"],
            config=data["config"],
            device=data["device"],
            precision=data["precision"],
            mode=data["mode"],
            is_baseline=data["is_baseline"],
            metrics=metrics,
            latency_samples_ms=list(data["latency_samples_ms"]),
            validation=data.get("validation"),
        )


@dataclass
class RunMetadata:
    """Provenance for a benchmark run."""

    timestamp: str
    git_sha: str | None
    seed: int
    hardware: dict[str, Any]
    library_versions: dict[str, Any]


@dataclass
class RunArtifact:
    """Top-level run artifact."""

    metadata: RunMetadata
    results: list[ResultRecord]
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metadata": asdict(self.metadata),
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunArtifact:
        found = data.get("schema_version")
        if found != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported artifact schema version {found!r}; "
                f"this build understands version {SCHEMA_VERSION}."
            )
        metadata = RunMetadata(**data["metadata"])
        results = [ResultRecord.from_dict(r) for r in data["results"]]
        return cls(metadata=metadata, results=results, schema_version=found)

    def results_for(self, model: str) -> list[ResultRecord]:
        """Return the result records belonging to a single model."""
        return [r for r in self.results if r.model == model]

    @property
    def models(self) -> list[str]:
        """Return the distinct model keys present in the artifact, in order."""
        seen: list[str] = []
        for record in self.results:
            if record.model not in seen:
                seen.append(record.model)
        return seen


def save_artifact(artifact: RunArtifact, path: str | Path) -> Path:
    """Write an artifact to disk as indented JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_artifact(path: str | Path) -> RunArtifact:
    """Load and validate an artifact from disk."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunArtifact.from_dict(data)
