# Methodology

This document is the compass for how the harness measures, validates, and
compares. It states the principles that tie the pieces together and points to a
dedicated document for each stage, where the full detail lives. The
[README](../README.md) carries the one-paragraph summary.

## What the harness does, and why it is split this way

The harness benchmarks a small set of reference models across a matrix of
execution modes and precisions, then asks two independent questions of every
configuration: is it still numerically correct, and is it still fast?

The two questions are kept deliberately separate because they fail for different
reasons and are trustworthy under different conditions. Numerical correctness is
hardware independent and is treated as a hard gate. Performance is hardware
sensitive and is gated only where the hardware is controlled. Each stage of the
pipeline is documented on its own so that the reasoning behind it can be read
without wading through the others.

## The reference point

Every comparison in the harness is defined against a single configuration: fp32
eager, on the same model. It is the baseline against which both numerical
validation and regression detection are measured. A meaningful comparison against
a fixed baseline has one precondition — the inputs must be identical from run to
run — which is why deterministic input generation is treated as part of the
benchmark procedure rather than an afterthought.

## The pipeline, stage by stage

- **[Benchmarking](benchmarking.md)** — the configuration matrix, how each
  configuration is timed and measured per device, and how the deterministic input
  data is generated, including the caveats that synthetic inputs carry.
- **[Numerical validation](numerical-validation.md)** — how non-baseline outputs
  are compared against the fp32 eager baseline, the four comparison metrics and
  the per-precision tolerances, and when and how an exact bit-wise comparison
  would be used instead of a tolerance-based one.
- **[Divergence localisation](divergence-localisation.md)** — capturing
  intermediate activations to find where divergence first enters and how it
  propagates through depth, including the observer effect that hooks introduce
  under `torch.compile`.
- **[Regression detection](regression-detection.md)** — treating latency as a
  distribution and flagging regressions with a statistical test plus a practical
  margin.
- **[Continuous integration and artifacts](ci-and-artifacts.md)** — how the two
  checks are gated differently in CI, and the schema-versioned artifact that the
  commands compose through.

## How the gating differs, in one place

The split above mirrors a single design decision that recurs throughout:

- **Numerical correctness is a hard gate.** Tolerance violations are host
  independent, so any divergence beyond the per-precision tolerances fails the
  build. See [numerical validation](numerical-validation.md).
- **Performance regression is report-only on shared runners.** Regression
  detection assumes a stable latency distribution, which does not hold on shared
  CI runners. The comparison is published but does not fail the build there; on
  dedicated hardware, strict gating is restored. See
  [regression detection](regression-detection.md) and
  [continuous integration](ci-and-artifacts.md).
