# Numerical validation

How the harness decides whether a configuration is still numerically correct.
This is one stage of the [methodology](methodology.md); it consumes the outputs
produced by [benchmarking](benchmarking.md) and shares its baseline with
[regression detection](regression-detection.md).

The question this stage answers is deliberately framed as *how close* two outputs
are, not *whether they are identical*. The reasoning behind that choice — and the
circumstances under which an exact, bit-wise comparison is the right tool instead
— is the subject of the second half of this document.

## What is compared, and against what

Every non-baseline configuration is compared against the fp32 eager output of the same
model. The comparison is implemented in `compare_outputs` (`src/mlrw/validate.py`), which
upcasts both tensors to fp32, checks that their shapes match, and returns four scalar
metrics.

## Comparison metrics

`compare_outputs` reduces the element-wise difference and the per-row geometry to four
numbers (`src/mlrw/validate.py:61-80`):

- **Maximum absolute difference** (`max_abs_diff`) and **mean absolute difference**
  (`mean_abs_diff`), computed from `diff = (base - cand).abs()` as `float(diff.max())`
  and `float(diff.mean())`. The maximum is the worst single element; the mean is the
  typical element. They are reported together because they answer different questions: a
  large max with a small mean is a localised excursion, while a large mean is a
  pervasive shift.
- **Cosine similarity** (`cosine_similarity`), averaged per row, capturing whether the
  output points in the same direction.
- **Top-k agreement** (`top_k_agreement`, k = 5), the fraction of overlapping top-k
  indices per row, capturing whether the ranking is preserved.

These four names are exactly the keys written under each result's `validation` block in
the JSON artifact, so the metric a number refers to is unambiguous when reading saved
results. For example, the headline MPS finding's divergence is recorded as
`max_abs_diff` — a **maximum** absolute difference, not a mean — and the corresponding
`mean_abs_diff` in the same record is roughly an order of magnitude smaller.

## Tolerances

Tolerances are defined per precision:

- **fp32** is held tight. Compiled fp32 is expected to match eager fp32 to within
  floating-point noise, so the absolute-difference gate is small.
- **bf16** relaxes the absolute-difference gate and is judged mainly by direction
  (cosine) and ranking (top-k agreement), because a reduced mantissa makes large
  absolute differences expected rather than alarming.

A configuration passes when every defined threshold holds. The check is hardware
independent, which is why continuous integration treats it as a hard gate (see
[continuous integration](ci-and-artifacts.md)).

## Why tolerance-based, not exact

The metrics above are all *approximate*: they ask how far apart two tensors are, with a
threshold that depends on the precision being judged. This is the correct default for
numerical validation of floating-point compute, for three reasons:

1. **Floating-point arithmetic is not associative.** A legitimate fp32 path can produce a
   result that differs from another legitimate fp32 path in the last few bits purely
   because operations were fused, reordered, or vectorised differently. An exact-equality
   gate would flag these as failures even though nothing is wrong.
2. **Reduced precision is expected to differ.** For bf16 the whole point is that the
   output is *approximately* the same; an exact comparison is meaningless there.
3. **The interesting failures are about magnitude.** "How wrong is it, and does it still
   point the right way?" is the question that distinguishes a harmless rounding difference
   from a configuration that silently dropped precision. A tolerance with companion
   direction/ranking metrics answers it; a boolean equal/not-equal does not.

## Exact (bit-wise) comparison

There is, nonetheless, one place in the codebase where equality is checked *exactly*, and
it is instructive about when exactness is the right instrument. The activation-capture
machinery asserts that the final block boundary and the model's returned final hidden
state are the same tensor, byte for byte (`tests/test_localize.py:90`):

```python
assert torch.equal(store["block_1"], store["last_hidden_state"])
```

`torch.equal` returns `True` only if two tensors have the same shape and every element is
bit-identical. That is the appropriate tool here precisely because the claim being tested
is structural — "these two capture points are literally the same tensor" — and not
numerical. There is no tolerance to choose because no arithmetic separates the two values;
they either are the same object's contents or they are not.

The same exact-equality idea appears in the investigation behind the headline MPS finding,
where a compiled fp32 output on MPS was observed to be **bit-identical to the bf16 output**
(`MPS compile_fp32 == MPS eager_bf16: True`, recorded in
[findings.md](findings.md#headline-finding-torchcompile-on-mps-computes-fp32-in-reduced-precision)).
It is worth being precise about the status of that statement: it is a *manual
experimental observation*, produced by an ad-hoc exact comparison while isolating the
cause, and it is reported in the findings narrative — it is **not** part of the validation
gate. The harness never gates on bit-equality between two configurations; the gate is the
tolerance-based comparison described above. The bit-identity observation is what made the
diagnosis unambiguous (an fp32 output that equals the bf16 output to the bit cannot be
genuinely full precision), but the harness flags the configuration through its
`max_abs_diff` against the fp32 baseline exceeding the fp32 tolerance.

### When a bit-wise comparison is the right tool

Exact equality is the right instrument whenever the property under test is itself exact,
rather than approximate:

- **Structural identity.** Two references that should be the same tensor (the
  `torch.equal` test above), or a tensor that should survive a serialise/deserialise round
  trip unchanged.
- **Determinism and reproducibility guarantees.** If the harness claims a fixed seed makes
  a run *exactly* reproducible on the same device and build, the way to verify that claim
  is to re-run and assert bit-equality, not to assert closeness — a tolerance would hide a
  determinism bug.
- **Caching and content-addressing.** Any time a tensor's bytes are hashed or used as a
  cache key, "equal" must mean byte-equal, because that is what the cache will mean by it.
- **Detecting silent kernel substitution.** When the difference of interest is *exactly
  zero* — proving that two paths computed the identical thing, as in the bf16-substitution
  diagnosis — exactness is the cleanest evidence, since a tiny non-zero tolerance could
  mask a genuine match or admit a near-match as if it were one.

### How to express it, and its limits

If the harness needed an exact gate, the building blocks are direct:

- **Whole-tensor exact equality:** `torch.equal(a, b)` (shape *and* every element), as in
  the existing test.
- **Exact zero difference:** `(a - b).abs().max().item() == 0`, when a single scalar
  summary of "no element differs at all" is wanted alongside the existing `max_abs_diff`
  pipeline. This reuses the reduction already computed in `compare_outputs`
  (`src/mlrw/validate.py:61-62`) and in the per-layer `_point_metrics`
  (`src/mlrw/localize.py:213-214`); the headline finding's probe also takes the same
  maximum-only reduction (`src/mlrw/localize.py:296-297`).
- **Byte / hash equality:** hashing the tensor's underlying bytes (after moving to CPU and
  fixing dtype and memory layout) when comparing against a stored fingerprint rather than
  a live tensor.

The limits matter as much as the mechanics. Exact equality is only meaningful **same
device, same build, same dtype, with determinism enabled** — the conditions
[benchmarking](benchmarking.md) already establishes through `set_determinism`. Across
devices it is effectively never attainable: MPS eager fp32 and CPU fp32 agree only to
about `7.6e-6` for genuinely correct computations, so an exact cross-device gate would
fail on correct code. This is exactly why the production gate is tolerance-based and an
exact comparison is reserved for the narrow, same-context cases above. A bit-wise check is
a scalpel for structural and determinism claims, not a substitute for the numerical
tolerance that validation relies on.
