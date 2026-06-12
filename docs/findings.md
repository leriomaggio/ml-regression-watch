# Findings and experimentation

This document records the experiments run with the harness and what they show. The
[README](../README.md) summarises the takeaways; the full narrative, tables, and the
investigation behind the headline finding are here. See [methodology.md](methodology.md)
for how the numbers are measured.

## Setup

All results below were produced on a single machine:

- **Hardware:** Apple M3 Pro, 12 logical cores. CPU and MPS (Metal) backends. No CUDA.
- **Software:** Python 3.11, PyTorch 2.12, torchvision 0.27, Transformers 4.57, with seed 1234.
- **Models:** torchvision ResNet-18 (convolutional, vision) and Hugging Face DistilBERT
  (transformer, text), batch size 8.
- **Configurations:** `eager_fp32` (baseline), `eager_bf16`, `compile_fp32`, `compile_bf16`.
- **Measurement:** 5 warmup iterations, then 8 repeat blocks of 20 timed iterations each.

Absolute latency and memory depend on the host, so the numbers below are illustrative.
They should not be compared against results from a different machine; the comparisons
within a single run, and the relative behaviour across configurations, are what carry
over.

## Latency

### CPU

![Median latency by model and configuration, CPU](latency_comparison_cpu.png)

| Model      | Config         | Median (ms) | p90 (ms) | Throughput (items/s) |
| ---        | ---            | ---:        | ---:     | ---:                 |
| resnet18   | `eager_fp32`   | 51.25       | 55.44    | 156.1                |
| resnet18   | `compile_fp32` | 47.04       | 47.70    | 170.1                |
| resnet18   | `compile_bf16` | 246.99      | 264.45   | 32.4                 |
| resnet18   | `eager_bf16`   | 1422.25     | 1566.37  | 5.6                  |
| distilbert | `eager_fp32`   | 48.36       | 50.49    | 165.4                |
| distilbert | `compile_fp32` | 46.79       | 47.34    | 171.0                |
| distilbert | `compile_bf16` | 254.71      | 263.80   | 31.4                 |
| distilbert | `eager_bf16`   | 253.85      | 262.74   | 31.5                 |

Observations:

- **Eager bf16 on CPU is very slow.** For ResNet-18 it is about 28 times slower than fp32
  (1422 ms versus 51 ms). The bf16 elementwise and convolution kernels on this CPU are
  not optimised, so reduced precision costs rather than saves time.
- **Compilation recovers, and slightly beats, fp32 speed** (`compile_fp32` is the fastest
  CPU configuration for both models).
- **bf16 is only worth compiling.** `compile_bf16` is far faster than `eager_bf16` but
  still slower than either fp32 configuration on CPU.

### MPS

![Median latency by model and configuration, MPS](latency_comparison_mps.png)

| Model      | Config         | Median (ms) | p90 (ms) | Throughput (items/s) |
| ---        | ---            | ---:        | ---:     | ---:                 |
| resnet18   | `eager_fp32`   | 21.38       | 23.59    | 374.2                |
| resnet18   | `compile_fp32` | 16.18       | 18.17    | 494.3                |
| resnet18   | `compile_bf16` | 16.58       | 18.82    | 482.6                |
| resnet18   | `eager_bf16`   | 65.32       | 90.51    | 122.5                |
| distilbert | `eager_fp32`   | 23.85       | 26.67    | 335.5                |
| distilbert | `compile_fp32` | 57.75       | 64.06    | 138.5                |
| distilbert | `compile_bf16` | 58.53       | 63.42    | 136.7                |
| distilbert | `eager_bf16`   | 15.99       | 17.12    | 500.4                |

Observations:

- **MPS roughly doubles eager fp32 throughput over CPU** (ResNet-18 51 ms to 21 ms,
  DistilBERT 48 ms to 24 ms).
- **Compilation helps the vision model and hurts the transformer.** `compile_fp32` is the
  fastest ResNet-18 configuration, but for DistilBERT compilation is about 2.4 times
  slower than eager (58 ms versus 24 ms). torch.compile is not a universal speed-up;
  whether it pays off is model and backend dependent, which is exactly why a benchmark
  matrix is worth running rather than assuming.
- **The fastest DistilBERT configuration on MPS is `eager_bf16`**, the opposite of CPU.

## Numerical validation

### CPU: everything within tolerance

| Model      | Config         | Max abs diff | Mean abs diff | Cosine  | Top-5 agree | Status |
| ---        | ---            | ---          | ---           | ---     | ---         | ---    |
| resnet18   | `compile_fp32` | 4.530e-06    | 6.608e-07     | 1.00000 | 1.000       | PASS   |
| resnet18   | `compile_bf16` | 4.133e-02    | 7.786e-03     | 0.99999 | 0.950       | PASS   |
| resnet18   | `eager_bf16`   | 3.026e-02    | 5.986e-03     | 0.99999 | 0.950       | PASS   |
| distilbert | `compile_fp32` | 4.172e-06    | 2.874e-07     | 1.00000 | 1.000       | PASS   |
| distilbert | `compile_bf16` | 3.740e-02    | 1.926e-03     | 0.99998 | 0.990       | PASS   |
| distilbert | `eager_bf16`   | 1.312e-02    | 1.842e-03     | 0.99998 | 0.990       | PASS   |

On CPU, compiled fp32 is numerically identical to eager fp32 to within floating-point
noise (about 1e-6), and bf16 diverges as expected but preserves direction and ranking.

### MPS: compiled fp32 fails the fp32 tolerance

| Model      | Config         | Max abs diff | Mean abs diff | Cosine  | Top-5 agree | Status |
| ---        | ---            | ---          | ---           | ---     | ---         | ---    |
| resnet18   | `compile_fp32` | 3.507e-02    | 6.338e-03     | 0.99999 | 0.975       | FAIL   |
| resnet18   | `compile_bf16` | 3.507e-02    | 6.338e-03     | 0.99999 | 0.975       | PASS   |
| resnet18   | `eager_bf16`   | 3.507e-02    | 6.338e-03     | 0.99999 | 0.975       | PASS   |
| distilbert | `compile_fp32` | 1.201e-02    | 1.785e-03     | 0.99998 | 0.994       | FAIL   |
| distilbert | `compile_bf16` | 1.201e-02    | 1.785e-03     | 0.99998 | 0.994       | PASS   |
| distilbert | `eager_bf16`   | 2.823e-02    | 1.877e-03     | 0.99998 | 0.991       | PASS   |

`compile_fp32` fails on both models because its absolute difference from the fp32 eager
baseline is far larger than fp32 should produce. The bf16 configurations show the same
divergence but pass, because the bf16 tolerance expects it.

## Headline finding: torch.compile on MPS computes fp32 in reduced precision

The MPS validation table shows a suspicious pattern: for each model, `compile_fp32` and
`compile_bf16` report exactly the same divergence. That is not what a genuine fp32 path
would do. The following experiment isolates the cause by running independent models and
comparing the raw outputs.

```
eager_fp32 (run twice)            max abs diff: 0.0          (deterministic)
CPU fp32   vs MPS eager_fp32      max abs diff: 7.6e-06      (agree: MPS eager fp32 is correct)
MPS eager_fp32 vs MPS compile_fp32 max abs diff: 0.0350695   (compiled fp32 diverges)
MPS eager_fp32 vs MPS eager_bf16   max abs diff: 0.0350695   (same divergence as bf16)
MPS compile_fp32 == MPS eager_bf16: True                     (bit-identical)
```

The conclusion is unambiguous:

- MPS **eager** fp32 matches CPU fp32 to 7.6e-6: the eager path is genuinely
  full precision.
- MPS **compiled** fp32 output is **bit-identical to bf16** and diverges from eager fp32
  by about 0.035.

In other words, `torch.compile` on the MPS backend computes a nominally full-precision
configuration in reduced precision. A user who selects `compile_fp32` for accuracy on MPS
does not get the precision they asked for.

The harness flags this as a tolerance violation on MPS, which is the intended behaviour.
A configuration that claims fp32 but does not deliver it is exactly the class of issue
numerical validation exists to catch, and it is invisible to a latency benchmark alone.
The cosine similarity remains 0.99999 and top-k agreement remains high, so the outputs
are directionally fine; whether the precision loss matters depends on the application,
which is why the harness reports the magnitude rather than only a pass or fail.

## Regression detection example

To illustrate a detected regression, an 18 percent slowdown was injected into one
configuration of a copy of a run, then compared against the original as baseline:

| Model      | Config         | Baseline ms | Current ms | Delta  | p-value | Effect | Status     |
| ---        | ---            | ---:        | ---:       | ---:   | ---:    | ---:   | ---        |
| resnet18   | `eager_fp32`   | 51.17       | 51.17      | +0.0%  | 0.521   | +0.00  | PASS       |
| resnet18   | `compile_bf16` | 248.32      | 248.32     | +0.0%  | 0.521   | +0.00  | PASS       |
| resnet18   | `compile_fp32` | 47.09       | 47.09      | +0.0%  | 0.521   | +0.00  | PASS       |
| resnet18   | `eager_bf16`   | 1441.46     | 1441.46    | +0.0%  | 0.521   | +0.00  | PASS       |
| distilbert | `eager_fp32`   | 48.23       | 48.23      | +0.0%  | 0.521   | +0.00  | PASS       |
| distilbert | `compile_bf16` | 254.36      | 254.36     | +0.0%  | 0.521   | +0.00  | PASS       |
| distilbert | `compile_fp32` | 46.75       | 55.17      | +18.0% | 0.000   | +1.00  | REGRESSION |
| distilbert | `eager_bf16`   | 253.80      | 253.80     | +0.0%  | 0.521   | +0.00  | PASS       |

Only the affected configuration is flagged. The unchanged configurations report a
p-value of 0.521 and zero effect size, well short of both the significance threshold and
the relative margin.

## Caveats

- Numbers are single-machine and host dependent. Treat them as relative, not absolute.
- The MPS finding is specific to this PyTorch version and backend; it is the kind of
  behaviour that can change between releases, which is itself an argument for running the
  validation continuously.
- On shared CI runners, performance regression detection is report-only because latency
  variance between runs is several-fold. See [methodology.md](methodology.md) for the
  gating rationale.
