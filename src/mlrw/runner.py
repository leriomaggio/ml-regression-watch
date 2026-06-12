"""Benchmark runner.

The runner applies an execution configuration to a model and measures its forward
pass. Each measurement discards warmup iterations, then records per-iteration
latencies. The procedure is repeated several times so that the regression test has a
sample of independent median-latency measurements rather than a single number.
"""

from __future__ import annotations

import contextlib
import gc
import time
from dataclasses import dataclass

import torch

from .artifacts import Metrics
from .config import Device, ExecConfig, Mode, Precision
from .models import extract_logits

# Mapping from the precision enum to the torch dtype used in the autocast context.
_AUTOCAST_DTYPE = {
    Precision.BF16: torch.bfloat16,
}


@dataclass
class ConfigResult:
    """In-memory result of benchmarking one configuration.

    ``logits`` is kept in memory for numerical validation and is never serialised
    directly; only the derived comparison metrics are stored in the artifact.
    """

    config: ExecConfig
    metrics: Metrics
    latency_samples_ms: list[float]
    logits: torch.Tensor
    compile_failed: bool = False


def _autocast(device: Device, precision: Precision):
    """Return an autocast context for reduced precision, or a no-op otherwise."""
    dtype = _AUTOCAST_DTYPE.get(precision)
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.value, dtype=dtype)


def _prepare_model(model: torch.nn.Module, config: ExecConfig) -> tuple[torch.nn.Module, bool]:
    """Move a model to the target device and apply the execution mode.

    Returns the prepared model and a flag indicating whether compilation was
    requested but failed. A compilation failure is reported rather than raised so
    that the remaining configurations still run.
    """
    model = model.to(config.device.value)
    if config.mode is Mode.COMPILE:
        try:
            compiled = torch.compile(model)
            return compiled, False
        except Exception:
            return model, True
    return model, False


def _move_inputs(inputs: dict[str, torch.Tensor], device: Device) -> dict[str, torch.Tensor]:
    return {key: value.to(device.value) for key, value in inputs.items()}


def _synchronize(device: Device) -> None:
    if device is Device.CUDA:
        torch.cuda.synchronize()
    elif device is Device.MPS:
        torch.mps.synchronize()


def _peak_memory_mb(device: Device, baseline_rss: int, mps_peak_bytes: int = 0) -> float:
    """Return peak memory in megabytes for the device.

    On CUDA the peak allocated device memory is reported. MPS exposes no peak
    counter, so the maximum of the current allocation sampled across iterations is
    used. On CPU the resident set size delta relative to a baseline is used as a
    portable approximation.
    """
    if device is Device.CUDA:
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    if device is Device.MPS:
        return mps_peak_bytes / (1024 * 1024)
    rss = _current_rss()
    return max(rss - baseline_rss, 0) / (1024 * 1024)


def _current_rss() -> int:
    """Return the current resident set size in bytes."""
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports ru_maxrss in bytes; Linux reports kilobytes.
    import sys

    if sys.platform == "darwin":
        return usage
    return usage * 1024


def benchmark(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    config: ExecConfig,
    *,
    warmup: int = 5,
    iters: int = 20,
    repeats: int = 5,
) -> ConfigResult:
    """Benchmark a model under a configuration.

    The model is prepared once, warmed up, and then timed across ``repeats`` blocks
    of ``iters`` iterations each. The median latency of each block is recorded as one
    sample; the reported median and p90 latency summarise all individual iterations.
    """
    device = config.device
    prepared, compile_failed = _prepare_model(model, config)
    device_inputs = _move_inputs(inputs, device)

    if device is Device.CUDA:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    elif device is Device.MPS:
        torch.mps.empty_cache()
    gc.collect()
    baseline_rss = _current_rss()

    all_latencies_ms: list[float] = []
    block_medians_ms: list[float] = []
    reference_logits: torch.Tensor | None = None
    mps_peak_bytes = 0

    with torch.inference_mode(), _autocast(device, config.precision):
        for _ in range(warmup):
            prepared(**device_inputs)
        _synchronize(device)

        for _ in range(repeats):
            block_latencies_ms: list[float] = []
            for _ in range(iters):
                start = time.perf_counter()
                output = prepared(**device_inputs)
                _synchronize(device)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                block_latencies_ms.append(elapsed_ms)
                if reference_logits is None:
                    reference_logits = extract_logits(output).detach().float().cpu()
                if device is Device.MPS:
                    # Sampled outside the timed region so it does not inflate latency.
                    mps_peak_bytes = max(mps_peak_bytes, torch.mps.current_allocated_memory())
            all_latencies_ms.extend(block_latencies_ms)
            block_medians_ms.append(_median(block_latencies_ms))

    assert reference_logits is not None
    batch_size = next(iter(device_inputs.values())).shape[0]
    median_ms = _median(all_latencies_ms)
    metrics = Metrics(
        median_latency_ms=median_ms,
        p90_latency_ms=_percentile(all_latencies_ms, 90),
        throughput_items_per_s=(batch_size / (median_ms / 1000.0)) if median_ms > 0 else 0.0,
        peak_memory_mb=_peak_memory_mb(device, baseline_rss, mps_peak_bytes),
    )
    return ConfigResult(
        config=config,
        metrics=metrics,
        latency_samples_ms=block_medians_ms,
        logits=reference_logits,
        compile_failed=compile_failed,
    )


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile, matching numpy's default method."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
