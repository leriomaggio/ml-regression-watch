# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A `--fail-on-regression / --no-fail-on-regression` option on `mlrw compare`. Strict
  gating remains the default for dedicated hardware; the continuous integration
  workflow uses the non-failing mode because shared runners have high latency variance,
  and publishes the regression report to the job summary instead.

## [0.1.0]

### Added

- Benchmark runner for ResNet-18 and DistilBERT across the eager and compiled modes
  at fp32 and bf16 precision, with warmup, per-iteration latency, throughput, and
  peak memory measurement.
- Versioned JSON run artifacts capturing run metadata, hardware, and library versions.
- Numerical correctness validation against the fp32 eager baseline, with per-precision
  tolerances and a Markdown divergence report.
- Statistical regression detection using the Mann-Whitney U test with a configurable
  relative margin and a rank-biserial effect size, with a Markdown report.
- Command-line interface with `run`, `validate`, `compare`, `update-baseline`, and
  `plot` commands.
- Continuous integration workflow that benchmarks, validates, compares against a
  committed baseline, and uploads reports as artifacts, plus a manual workflow to
  regenerate the baseline.
- Test suite covering configuration, artifact schema, comparison metrics, tolerance
  logic, and statistical regression detection with injected synthetic regressions.

[Unreleased]: https://github.com/leriomaggio/ml-regression-watch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/leriomaggio/ml-regression-watch/releases/tag/v0.1.0
