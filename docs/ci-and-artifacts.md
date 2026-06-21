# Continuous integration and artifacts

How the two checks are gated in CI, and the artifact that every command composes
through. This is one stage of the [methodology](methodology.md); the checks
themselves are [numerical validation](numerical-validation.md) and
[regression detection](regression-detection.md).

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
