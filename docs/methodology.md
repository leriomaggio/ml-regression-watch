# Methodology

This document explains how the harness measures, validates, and compares. The
[README](../README.md) carries the summary; the detail lives here.

## Configuration matrix

A run benchmarks each model across the matrix of execution modes and precisions:

| Mode    | Precision | Configuration name      |
| ---     | ---       | ---                     |
| eager   | fp32      | `eager_fp32` (baseline) |
| eager   | bf16      | `eager_bf16`            |
| compile | fp32      | `compile_fp32`          |
| compile | bf16      | `compile_bf16`          |

The matrix is built from orthogonal device, precision, and mode axes
(`src/mlrw/config.py`). The fp32 eager configuration is the baseline against which both
numerical validation and regression detection are defined. Adding a device or precision
is a single enum entry: the runner resolves device-specific synchronisation, autocast,
and memory accounting behind a uniform interface.

Devices: CPU, CUDA, and Apple MPS are supported. `--device auto` selects an accelerator
when one is visible, in the order CUDA, then MPS, then CPU.

## Benchmark procedure

For each configuration the runner:

1. Moves the model to the device and applies the mode. Compiled modes wrap the model
   with `torch.compile`; if compilation fails, the configuration falls back to eager and
   the artifact records the fallback rather than aborting the run.
2. Runs `warmup` forward passes that are discarded, then times `repeats` blocks of
   `iters` passes each. Reduced precision is applied through an autocast context, so the
   fp32 weights are never mutated and the baseline stays intact.
3. Records, per configuration: median and p90 latency over all timed iterations,
   throughput in items per second, and peak memory. The median latency of each repeat
   block is stored as one sample, giving the distribution that regression detection
   consumes.

Device-specific details:

- **Timing** uses `time.perf_counter` with a device synchronisation (`torch.cuda` or
  `torch.mps`) before the elapsed time is read, so asynchronous work is fully accounted.
- **Peak memory** uses `torch.cuda.max_memory_allocated` on CUDA. MPS exposes no peak
  counter, so the maximum of `torch.mps.current_allocated_memory` sampled across
  iterations (outside the timed region) is used. CPU uses the resident-set-size delta as
  a portable approximation.

Inputs are generated from fixed seeds, so repeated runs produce identical tensors. This
is a precondition for comparing outputs across configurations.

## Numerical validation

Every non-baseline configuration is compared against the fp32 eager output of the same
model. Four metrics summarise each comparison:

- **Maximum absolute difference** and **mean absolute difference** of the output tensors.
- **Cosine similarity**, averaged per row, capturing whether the output points in the
  same direction.
- **Top-k agreement** (k = 5), the fraction of overlapping top-k indices per row,
  capturing whether the ranking is preserved.

Tolerances are defined per precision:

- **fp32** is held tight. Compiled fp32 is expected to match eager fp32 to within
  floating-point noise, so the absolute-difference gate is small.
- **bf16** relaxes the absolute-difference gate and is judged mainly by direction
  (cosine) and ranking (top-k agreement), because a reduced mantissa makes large
  absolute differences expected rather than alarming.

A configuration passes when every defined threshold holds. The check is hardware
independent, which is why continuous integration treats it as a hard gate.

## Regression detection

Regression detection treats the per-repeat latency measurements of a configuration as a
sample rather than a single number, and compares the current sample against the stored
baseline sample with a one-sided Mann-Whitney U test (`scipy.stats.mannwhitneyu`,
`alternative="greater"`).

A configuration is flagged as a regression only when both conditions hold:

1. The test is significant (p < 0.05): the current latencies are stochastically greater
   than the baseline latencies.
2. The relative increase in median latency exceeds a configurable margin (default 5
   percent).

Requiring both avoids flagging differences that are statistically detectable but
practically irrelevant. The report includes the rank-biserial effect size, derived from
the same U statistic and bounded in [-1, 1], alongside the relative change.

## Continuous integration

The `ci` workflow installs the package, lints, runs the unit tests, benchmarks on CPU
with reduced repeats, validates, compares against the committed baseline, and uploads the
artifact and reports. The two checks are gated differently, by design:

- **Numerical correctness is a hard gate.** Tolerance violations are host independent, so
  the job fails on any divergence beyond the per-precision tolerances.
- **Performance regression is report-only on shared runners.** Regression detection
  assumes a stable latency distribution, which holds on dedicated hardware but not on
  shared GitHub runners, where the same code can vary several-fold between runs. The
  comparison runs with `--no-fail-on-regression`: the report is produced and published to
  the job summary, but runner noise does not fail the build. On dedicated hardware,
  removing the flag restores strict gating. A microbenchmark regression gate is only
  meaningful on controlled hardware.

The committed baseline must be produced on the same runner image as the comparison, or
hardware differences would masquerade as regressions. The `baseline` workflow
(`workflow_dispatch`) regenerates it on the runner and uploads it for review before it is
committed. Until a baseline exists, the comparison step reports that none was found and
the job passes.

## Artifacts

Each run is recorded as a schema-versioned JSON artifact containing run metadata
(timestamp, git SHA, hardware, library versions, seed) and one record per
(model, configuration) with its metrics, latency samples, and validation result. The
commands compose through this artifact: `run` produces it; `validate`, `compare`, and
`plot` consume it. Each step is therefore independently runnable in a pipeline, and the
artifact is the durable, reviewable record of a run.
