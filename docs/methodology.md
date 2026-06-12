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

## Divergence localisation

Numerical validation compares only final outputs. Divergence localisation captures
intermediate activations at eight depth boundaries of DistilBERT and reports where
divergence from the fp32 eager baseline first appears and how it propagates through
depth. Localisation is scoped to DistilBERT, where the transformer numerics are the
interesting case. Unlike the other commands it runs the model directly rather than
consuming a stored artifact, because intermediate activations are not part of the
artifact schema.

### Capture points

Forward hooks record eight activations in depth order: the embedding output, the output
of each of the six transformer blocks, and the final hidden state. Hooks are registered
on the embedding module and on each transformer block, resolved by attribute path, and
the final hidden state is read from the forward return value. For the head-less
`DistilBertModel` the sixth block boundary and the final hidden state are the same
tensor; both are kept so that the mechanism reads identically for a model that carries a
pre-classifier head.

Each activation is detached, moved to CPU, and upcast to fp32 inside the hook, so device
memory is released as the forward pass proceeds and every comparison happens in fp32. The
two activation tensors of a capture point are released as soon as their metrics are
computed, so the full baseline and target activation sets are not both held for every
point at once.

### Metrics

For each capture point four metrics summarise the divergence from the fp32 eager
baseline:

- **Maximum** and **mean absolute difference** of the activation tensors.
- **Cosine similarity**, averaged per row.
- **Relative error**: the Frobenius norm of the difference normalised by the baseline
  activation norm. This is the depth-comparable metric, because it is invariant to the
  changing scale of activations through the network, whereas absolute differences grow
  and shrink with the activation magnitude that each layer happens to produce.

First divergence is the earliest capture point whose relative error exceeds a
configurable threshold (default `1e-3`), reported explicitly.

### Hooks under torch.compile

Hooks and `torch.compile` interact in a way that determines what localisation can
honestly claim. On this build (PyTorch 2.12) hooks fire under compilation and every
capture point is recorded. Materialising an intermediate activation, however, forces a
graph break, and where the compiled whole-graph performs a reduced-precision substitution,
that graph break can suppress it. This was observed directly with Transformers 4.57.6 on
the MPS backend: a hooked compiled DistilBERT reports genuine fp32 activations, agreeing
with eager fp32 to about `6e-6` at every depth, while the same model run with no hooks
diverges from eager fp32 by about `1.2e-2`. Returning the activations through
`output_hidden_states` has the same effect, which confirms that the cause is the
materialisation of an intermediate value rather than hooks specifically. The substitution
is therefore a property of the fully fused whole-graph compilation, and any probe that
reads an intermediate value dissolves it.

This observer effect is real but not constant. Under the pinned Transformers 5.0.0rc3 the
DistilBERT substitution no longer occurs at all (the 5.0 attention refactor changes the
compiled graph; see [findings.md](findings.md#version-sensitivity-the-distilbert-substitution-depends-on-the-attention-path)),
so the hooked and un-hooked passes agree and the per-layer curve is faithful. The design
below is what makes the report correct under either version without assuming which one is
in use.

The consequence is a deliberate split:

- For **eager** configurations the per-layer curve is faithful. Autocast applies reduced
  precision without restructuring the graph, so the captured activations are exactly
  those the configuration computes.
- For **compiled** configurations the per-layer curve from hooks may reflect a
  graph-broken path rather than the production compiled path, and a flat noise-level curve
  on its own cannot distinguish a faithful configuration from one whose divergence the
  hooks have suppressed. To resolve that ambiguity rather than silently mislead, a compiled
  target additionally runs one un-hooked whole-model pass, and the report records the
  contrast between the hooked and un-hooked final divergence. A large contrast means
  probing dissolved a whole-graph substitution that no single capture point can therefore
  localise; agreement means probing did not alter the numerics and the per-layer curve is
  faithful. The report states which case holds.

This is the sound option among those available. There is no `torch._dynamo` configuration
that preserves both the hooks and the fused-graph numerics, because the two are mutually
exclusive by construction: the hook needs the intermediate value materialised and the
fusion needs it not to be. Reconstructing compiled depth-`k` prefixes was considered and
rejected, because it requires replicating the model's internal attention-mask
preparation, which is version specific and fragile.

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
