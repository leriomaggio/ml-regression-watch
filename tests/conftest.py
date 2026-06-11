"""Shared test fixtures.

The fixtures here avoid downloading or constructing the reference models. A tiny
deterministic network stands in for a real model when the runner itself is under
test, and synthetic artifacts are built directly so that the validation and
comparison logic can be tested without running a benchmark.
"""

from __future__ import annotations

import torch
from torch import nn

from mlrw.artifacts import Metrics, ResultRecord, RunArtifact, RunMetadata


class TinyNet(nn.Module):
    """A minimal network used to exercise the runner without heavy dependencies."""

    def __init__(self, in_features: int = 16, out_features: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def make_tiny_model() -> nn.Module:
    torch.manual_seed(0)
    model = TinyNet()
    model.eval()
    return model


def make_tiny_input() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {"x": torch.randn(4, 16)}


def make_metrics(median: float = 10.0) -> Metrics:
    return Metrics(
        median_latency_ms=median,
        p90_latency_ms=median * 1.1,
        throughput_items_per_s=1000.0 / median,
        peak_memory_mb=64.0,
    )


def make_record(
    *,
    model: str = "resnet18",
    config: str = "eager_fp32",
    precision: str = "fp32",
    mode: str = "eager",
    is_baseline: bool = False,
    samples: list[float] | None = None,
    validation: dict[str, float] | None = None,
) -> ResultRecord:
    samples = samples if samples is not None else [10.0, 10.1, 9.9, 10.2, 9.8]
    return ResultRecord(
        model=model,
        config=config,
        device="cpu",
        precision=precision,
        mode=mode,
        is_baseline=is_baseline,
        metrics=make_metrics(sum(samples) / len(samples)),
        latency_samples_ms=samples,
        validation=validation,
    )


def make_metadata() -> RunMetadata:
    return RunMetadata(
        timestamp="2026-01-01T00:00:00+00:00",
        git_sha="0" * 40,
        seed=1234,
        hardware={"platform": "test", "cpu_count": 1, "cuda_available": False},
        library_versions={"python": "3.11.0", "torch": "2.12.0"},
    )


def make_artifact(records: list[ResultRecord]) -> RunArtifact:
    return RunArtifact(metadata=make_metadata(), results=records)
