# Divergence localisation

Numerical validation compares only final outputs. Divergence localisation finds
*where* a divergence first enters the network and how it propagates through depth.
This is one stage of the [methodology](methodology.md); it builds on the same
fp32 eager baseline as [numerical validation](numerical-validation.md).

Localisation captures intermediate activations at eight depth boundaries of DistilBERT and
reports where divergence from the fp32 eager baseline first appears and how it propagates
through depth. Localisation is scoped to DistilBERT, where the transformer numerics are the
interesting case. Unlike the other commands it runs the model directly rather than
consuming a stored artifact, because intermediate activations are not part of the
artifact schema.

## Capture points

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

## Metrics

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

## Hooks under torch.compile

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
