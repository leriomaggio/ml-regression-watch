# Benchmarking

How the harness lays out the configuration matrix, measures each configuration,
and produces the deterministic inputs every measurement depends on. This is one
stage of the [methodology](methodology.md); the comparisons that consume these
measurements are described in [numerical validation](numerical-validation.md) and
[regression detection](regression-detection.md).

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

## Input data generation

Inputs are generated from a single fixed seed so that repeated runs produce identical
tensors. This is a precondition for comparing outputs across configurations: if the
inputs drifted between the baseline pass and a candidate pass, any output difference
would conflate an input change with the precision or mode change under test. The
generators and the seeding live in `src/mlrw/models.py`.

`set_determinism(seed=1234)` seeds the Python, NumPy, and Torch RNGs (including the CUDA
and MPS generators) and requests deterministic cuDNN kernels. It is called immediately
before every model load and every input construction, so the same tensors are produced
regardless of call order or device.

The two reference models are fed synthetic inputs of fixed shape:

- **ResNet-18** receives a pseudo-image batch `torch.randn(8, 3, 224, 224)` — Gaussian
  noise at standard ImageNet spatial dimensions. The model itself is randomly initialised
  (`weights=None`): the goal is to exercise the full convolutional compute graph, not to
  produce a correct classification, and determinism comes from the fixed seed rather than
  a pretrained checkpoint.
- **DistilBERT** receives `input_ids = torch.randint(0, 30522, (8, 64))` — uniformly
  random token ids over the `distilbert-base-uncased` vocabulary — together with an
  all-ones attention mask of the same shape. The tokenizer is not involved; token id
  tensors are produced directly. The model weights are the real pretrained checkpoint.

Batch sizes are deliberately small (8) to keep CPU runs fast while still driving the
model's characteristic compute profile.

### Caveats: how realistic is this?

The inputs are synthetic by design, and that design choice has limits worth stating
plainly:

- **Random token ids are not language.** They carry no syntax, no semantics, and a
  near-uniform token distribution, so the embedding and attention activations they induce
  do not match those of natural text. The numerics being exercised are real, but their
  *distribution* is not representative of a production text workload.
- **The attention mask is all ones.** No padding is ever present, so the masking and
  variable-length paths are never exercised. A divergence that only manifests with padded
  batches would not be caught.
- **Sequence length and batch are fixed** (64 and 8). Shape-dependent behaviour — kernel
  selection, fusion decisions, or precision substitutions that only appear at other
  shapes — is outside the matrix unless the shape is changed explicitly.
- **ResNet runs on Gaussian noise with random weights.** Pixel statistics and the learned
  filters of a real network both differ from this, so the activation magnitudes are
  representative of the architecture but not of any real image-plus-checkpoint pairing.

None of this undermines the harness's purpose, because it measures the *numerical and
timing behaviour of the compute graph*, not predictive quality, and for that purpose a
fixed, reproducible input is exactly what is wanted. The caveat to keep in mind is
quantitative: the per-precision tolerances and the divergence magnitudes reported
elsewhere were observed on this synthetic distribution, and the *magnitude* of a
divergence can shift on real data even when its *presence* does not. Where a finding
depends on a specific divergence figure, that figure should be read as characteristic of
these inputs rather than as a universal constant. Swapping in realistic inputs (real
tokenised text, real images) is a drop-in change to the input factories in
`src/mlrw/models.py` and would be the right move before quoting an absolute tolerance for
a production workload.
