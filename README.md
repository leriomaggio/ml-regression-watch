# Numerical Validation and Performance Regression Detection for ML Models

A small CLI-driven harness that benchmarks ML models across execution configurations,
validates that reduced-precision and compiled configurations stay numerically faithful
to a full-precision baseline, and detects performance regressions with a statistical
test rather than a fixed threshold.

## TL;DR

- **A compiled configuration can silently change precision.** On the Apple MPS backend,
  `torch.compile` runs ResNet-18 in reduced precision: the compiled fp32 output is
  bit-identical to bf16 and diverges from eager fp32 by about 0.035, while CPU and MPS
  eager fp32 agree to 7.6e-6. The harness catches this; a latency benchmark alone would
  not. Whether a given model triggers it depends on the exact compiled graph: DistilBERT
  showed the same substitution under Transformers 4.57.6, but the 5.0.0rc3 attention
  refactor removed it, and the harness flips that configuration from fail to pass on the
  dependency bump alone. See [docs/findings.md](docs/findings.md#headline-finding-torchcompile-on-mps-computes-fp32-in-reduced-precision).
- **Reduced precision is not automatically faster.** Eager bf16 on CPU is about 28 times
  slower than fp32 (unoptimised kernels); `torch.compile` helps ResNet-18 on MPS but
  makes DistilBERT about 2.4 times slower. Whether a configuration pays off is model and
  backend dependent.
- **Regression gating only means something on controlled hardware.** Detection uses a
  one-sided Mann-Whitney U test plus a relative margin; on shared CI runners, where
  latency varies several-fold between runs, it runs report-only while numerical
  validation stays a hard gate.

## What it does

- **Benchmarks** ResNet-18 and DistilBERT across `eager`/`compile` x `fp32`/`bf16` on
  CPU, CUDA, or Apple MPS, with warmup, median and p90 latency, throughput, and peak
  memory.
- **Validates** each configuration against the fp32 eager baseline (max and mean abs
  diff, cosine similarity, top-k agreement) with per-precision tolerances.
- **Detects regressions** against a stored baseline using a statistical test with an
  effect size, not a single-number threshold.
- **Localises divergence** through DistilBERT's depth with forward hooks, capturing eight
  activation boundaries to report where divergence from the fp32 eager baseline first
  appears and how it propagates. On MPS it traces eager bf16 amplifying with depth, and a
  per-target un-hooked probe distinguishes a faithful compiled path from one whose
  divergence the hooks themselves suppress.
- **Runs in CI** and is itself covered by a pytest suite.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

mlrw run --out artifacts/run.json --device auto          # benchmark the matrix
mlrw validate artifacts/run.json --report reports/validation.md   # numerical check
mlrw compare --baseline baselines/cpu_ci.json --current artifacts/run.json   # regressions
mlrw update-baseline --from artifacts/run.json --out baselines/cpu_ci.json   # promote a run
```

`--device auto` prefers an accelerator (CUDA, then MPS, then CPU); `cpu`, `cuda`, and
`mps` force a device.

## Configurations

| Mode    | Precision | Name                    |
| ---     | ---       | ---                     |
| eager   | fp32      | `eager_fp32` (baseline) |
| eager   | bf16      | `eager_bf16`            |
| compile | fp32      | `compile_fp32`          |
| compile | bf16      | `compile_bf16`          |

Built from orthogonal device, precision, and mode axes, so adding a device or precision
is a single enum entry.

## Results at a glance

Median latency by model and configuration (log scale, Apple M3 Pro). Full tables, the
numerical-divergence analysis, and the MPS precision investigation are in
[docs/findings.md](docs/findings.md).

CPU:

![Median latency by model and configuration, CPU](docs/latency_comparison_cpu.png)

MPS:

![Median latency by model and configuration, MPS](docs/latency_comparison_mps.png)

Divergence from the fp32 eager baseline through DistilBERT's depth (MPS, log scale). Eager
bf16 enters at the first block and amplifies; the compiled configurations sit at
floating-point noise in the hooked curve, and the un-hooked probe is what tells a faithful
path apart from one whose divergence the hooks suppress. The full analysis, including the
library-version sensitivity, is in [docs/findings.md](docs/findings.md#divergence-localisation-where-reduced-precision-enters-and-how-it-propagates).

![Divergence from the fp32 eager baseline by depth, MPS](docs/divergence_by_depth.png)

## How it works

- **Numerical validation** compares outputs to the fp32 eager baseline and gates per
  precision: fp32 tight, bf16 judged by direction and ranking. Hardware independent.
- **Regression detection** treats per-repeat latencies as a sample, applies a one-sided
  Mann-Whitney U test, and flags a configuration only when the slowdown is both
  significant (p < 0.05) and beyond a relative margin (default 5 percent).
- **CI** hard-fails on tolerance violations and runs regression detection report-only on
  shared runners.

Full detail in [docs/methodology.md](docs/methodology.md).

## Architecture

```
mlrw/
  config.py      Device / Precision / Mode axes and the ExecConfig matrix
  models.py      ResNet-18 and DistilBERT loaders with deterministic inputs
  runner.py      Warmup, timed loop, and latency / throughput / memory metrics
  artifacts.py   Versioned JSON schema with run, hardware, and library metadata
  validate.py    Pairwise numerical comparison against the fp32 eager baseline
  compare.py     Mann-Whitney U regression detection with effect size
  hardware.py    Hardware and library-version introspection
  localize.py    Per-layer divergence capture with forward hooks
  plotting.py    Latency comparison and divergence-by-depth plots
  reporting.py   Shared Markdown helpers
  cli.py         Typer commands: run / validate / compare / update-baseline / plot / localize
```

The commands compose through a single JSON artifact, so each step is independently
runnable in a pipeline.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

The suite under `tests/` covers configuration parsing, the artifact schema, the
comparison metrics, the tolerance logic, and the statistical detection on synthetic data
with injected regressions.

## Documentation

- [docs/findings.md](docs/findings.md) — experiments, results, and the MPS precision finding.
- [docs/methodology.md](docs/methodology.md) — how the harness measures, validates, and compares.

## License

Released under the MIT License. See [LICENSE](LICENSE).
