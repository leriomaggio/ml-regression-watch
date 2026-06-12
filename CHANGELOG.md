# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Version-stamped the MPS compiled-fp32 reduced-precision finding. The substitution is a
  backend behaviour that ResNet-18 still triggers, but the DistilBERT case depends on the
  attention path: it reproduced under Transformers 4.57.6 and no longer does under the
  pinned 5.0.0rc3, where compiled fp32 matches eager fp32 to about 6e-6 and passes
  validation. The findings, methodology, and README now document this across both versions
  rather than presenting the result as timeless.
- Restructured the documentation: the README now leads with the key findings and stays
  scannable, while the full experimentation narrative moved to `docs/findings.md` and the
  measurement and gating detail to `docs/methodology.md`. Added per-device latency charts
  (`docs/latency_comparison_cpu.png`, `docs/latency_comparison_mps.png`).

### Added

- Per-layer divergence localisation for DistilBERT. A new `mlrw localize` command captures
  activations at eight depth boundaries with forward hooks and reports where divergence
  from the fp32 eager baseline first appears, with a relative-error-versus-depth plot
  (`docs/divergence_by_depth.png`). Because hooks force a graph break that suppresses the
  MPS compiled reduced-precision substitution, compiled targets also run one un-hooked
  pass and the report records the contrast.
- Apple MPS as a supported device. `--device auto` now selects an accelerator in the
  order CUDA, MPS, CPU, and `--device mps` forces the MPS backend. The runner handles
  MPS synchronisation and memory accounting alongside the existing CPU and CUDA paths.
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
