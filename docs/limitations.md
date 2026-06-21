# Threats to validity

The findings produced by this harness, including the torch.compile / MPS
precision finding recorded in [findings.md](findings.md), are bounded by the
choice of inputs. The inputs are synthetic and fixed, which is the right
trade-off for a reproducible numerical probe, but it limits how far the measured
numbers generalise. This document records those limits so that each figure is
read with the correct scope. It is one stage of the [methodology](methodology.md);
the input generators themselves are described in [benchmarking.md](benchmarking.md).

## Where reduced precision enters each model

Reduced precision is applied through `torch.autocast`, not by casting the model.
The `_autocast` helper in `src/mlrw/runner.py` returns an autocast context for the
bf16 configurations and a null context otherwise, because `_AUTOCAST_DTYPE` maps
only the bf16 precision to a dtype. The fp32 eager baseline therefore runs with no
autocast at all. Under autocast the parameters stay fp32 and only the
autocast-eligible operations, matrix multiplication, convolution, and linear
layers, run in bf16, while operations such as normalisation stay fp32. This was
confirmed directly: an embedding lookup and a layer normalisation keep fp32
outputs under bf16 autocast, while a linear layer, a matrix multiplication, and a
convolution produce bf16 outputs.

The consequence is that reduced precision enters the two reference models at
different depths.

- ResNet-18 receives its input from `make_resnet_input`, which produces continuous
  standard-normal pixels with `torch.randn`. Convolution is autocast-eligible, so
  the pixel input is rounded to bf16 at the first convolution. The input signal
  itself is approximated, and that rounding is part of the measured divergence.
- DistilBERT receives its input from `make_distilbert_input`, which produces
  discrete integer token identifiers with `torch.randint` together with an
  attention mask of ones. Integer token identifiers are exact in every
  configuration, and an embedding lookup is not an autocast-eligible operation, so
  the embedding table stays fp32 and the embeddings are full precision. Reduced
  precision begins one step later, at the first eligible matrix multiplication in
  the attention and feed-forward projections, not at the input or the embedding.
  The divergence-by-depth table in [findings.md](findings.md#divergence-localisation-where-reduced-precision-enters-and-how-it-propagates)
  records this independently: the `embeddings` capture point shows a maximum
  absolute difference of `0.000e+00`, and divergence first appears at `block_0`.

Because of this, the two divergence figures do not measure the same thing. On MPS
under Transformers 4.57.6 the `compile_fp32` maximum absolute difference is
`3.507e-02` for ResNet-18 and `1.201e-02` for DistilBERT. These figures must not
be compared directly as a claim about which architecture is more
precision-sensitive, because the locus of precision loss differs between them: for
ResNet the input is already approximated, while for DistilBERT the input and the
embeddings are exact. The locus of precision loss is a confound.

## The inputs are synthetic

The inputs are not only fixed but also unstructured. The `randn` pixels are white
noise, and the `randint` token identifiers are uniform over the vocabulary,
whereas real images have spatial structure and real text follows a non-uniform,
Zipfian distribution. Because bf16 error is relative to the magnitude of the
values being rounded, and activation magnitudes depend on the structure of the
input, a divergence measured on random inputs need not transfer to real data.

## Scope: what this does and does not undermine

The core compile-path finding is a property of kernel selection in the compiled
graph. That selection is input-independent, so random, shape-controlled inputs are
a valid and arguably preferable probe for detecting the substitution: they are
reproducible, they need no data pipeline, and they isolate the compute path from
the data.

The synthetic inputs limit only the downstream claims:

- The real-world magnitude of the divergence. The reported figures are
  characteristic of these inputs, not of a production workload.
- Any accuracy or quality claim. There is no task and no ground truth, so the
  divergence cannot be expressed as a change in predictive quality.
- Cross-model sensitivity comparison, for the reason given above: the locus of
  precision loss differs between the two models.

## How the harness would be hardened

The following changes would lift the limits above without changing the core
method.

- Add real, representative inputs alongside the synthetic ones, so that magnitudes
  reflect a realistic workload.
- Magnitude-match or profile the synthetic inputs to reproduce a realistic dynamic
  range, so that the bf16 rounding error is representative even without real data.
- Attach a ground-truth task, so that divergence can be expressed as an accuracy
  delta rather than only a tensor distance.
