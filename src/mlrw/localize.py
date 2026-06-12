"""Per-layer divergence localisation.

Numerical validation compares only the final model output against the fp32 eager
baseline. This module localises divergence: it captures intermediate activations at
the boundaries of each transformer block and asks where in the network divergence
first appears and how it propagates through depth.

The capture mechanism is forward hooks. Eight capture points are recorded for
DistilBERT: the embedding output, the output of each of the six transformer blocks,
and the final hidden state. For the head-less ``DistilBertModel`` the sixth block
boundary and the final hidden state are the same tensor; both are kept so that the
mechanism reads the same way for a model that does carry a pre-classifier head.

Hooks interact with ``torch.compile`` in a way that matters here and is documented in
``docs/methodology.md``. Hooks fire under compilation, but materialising an
intermediate activation forces a graph break, and on the MPS backend that graph break
suppresses the reduced-precision substitution that compiled fp32 otherwise performs.
A hooked compiled run therefore reports genuine fp32 activations while the un-hooked
run diverges. The captured per-layer curve is faithful for eager configurations; for
compiled configurations the divergence is a whole-graph property that the act of
probing dissolves, so a compiled target additionally runs one un-hooked whole-model
pass and the report records the contrast.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
from torch import nn

from .config import BASELINE_MODE, BASELINE_PRECISION, Device, ExecConfig, Mode, Precision
from .models import extract_logits
from .reporting import markdown_table

# The eight capture points in depth order. ``block_5`` and ``last_hidden_state`` carry
# the same tensor for the head-less model; see the module docstring.
CAPTURE_POINTS: tuple[str, ...] = (
    "embeddings",
    "block_0",
    "block_1",
    "block_2",
    "block_3",
    "block_4",
    "block_5",
    "last_hidden_state",
)

# Capture points written by hooks on submodules. ``last_hidden_state`` is read from the
# forward return value rather than a hook, because it is the model output.
_HOOKED_POINTS: tuple[str, ...] = CAPTURE_POINTS[:-1]

# Relative error below this magnitude is floating-point noise rather than divergence.
DEFAULT_THRESHOLD = 1.0e-3

# Mapping from the precision enum to the torch dtype used in the autocast context,
# mirroring the runner so that capture and benchmarking apply reduced precision the
# same way.
_AUTOCAST_DTYPE = {
    Precision.BF16: torch.bfloat16,
}


@dataclass(frozen=True)
class LayerDivergence:
    """Divergence metrics at a single capture point."""

    point: str
    max_abs_diff: float
    mean_abs_diff: float
    cosine_similarity: float
    relative_error: float


@dataclass
class LocalizationReport:
    """Result of localising divergence of one configuration against the baseline."""

    model: str
    config: str
    device: str
    threshold: float
    layers: list[LayerDivergence]
    first_divergence: str | None
    probe_effect: dict[str, float] | None = None


def _hidden_from_output(output: object) -> torch.Tensor:
    """Reduce a module's forward output to the hidden-state tensor it carries.

    DistilBERT transformer blocks return a length-one tuple whose element is the
    hidden state when attention outputs are not requested. The embedding module
    returns the tensor directly.
    """
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"Cannot read a hidden-state tensor from output of type {type(output)!r}")


def distilbert_capture_targets(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Resolve the hooked capture points of a DistilBERT model.

    The boundaries are the embedding module and each of the six transformer blocks.
    The resolution is by attribute path, so a model that exposes the same structure is
    captured without further wiring.
    """
    targets: list[tuple[str, nn.Module]] = [("embeddings", model.embeddings)]
    for index, block in enumerate(model.transformer.layer):
        targets.append((f"block_{index}", block))
    return targets


def register_capture_hooks(
    targets: list[tuple[str, nn.Module]], store: dict[str, torch.Tensor]
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register forward hooks that record each target's output into ``store``.

    Activations are detached, moved to CPU, and upcast to fp32 inside the hook so that
    device memory is released as soon as the forward pass moves on and every later
    comparison happens in fp32.
    """
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for name, module in targets:

        def _hook(_module, _inputs, output, _name: str = name) -> None:
            store[_name] = _hidden_from_output(output).detach().to("cpu", torch.float32)

        handles.append(module.register_forward_hook(_hook))
    return handles


def _autocast(device: Device, precision: Precision):
    """Return an autocast context for reduced precision, or a no-op otherwise."""
    dtype = _AUTOCAST_DTYPE.get(precision)
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.value, dtype=dtype)


def capture_activations(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    config: ExecConfig,
    *,
    targets: list[tuple[str, nn.Module]] | None = None,
) -> dict[str, torch.Tensor]:
    """Capture the eight activations of one configuration.

    The model is moved to the device and hooks are registered before compilation, so
    that the hooks sit on the eager submodules and fire under a compiled wrapper. The
    final hidden state is read from the forward return value rather than a hook.
    """
    device = config.device
    model = model.to(device.value)
    if targets is None:
        targets = distilbert_capture_targets(model)

    store: dict[str, torch.Tensor] = {}
    handles = register_capture_hooks(targets, store)

    runnable: nn.Module = model
    if config.mode is Mode.COMPILE:
        # Reset so a previously compiled graph for the same code is not reused with a
        # different hook configuration, which would leave capture points unwritten.
        torch._dynamo.reset()
        runnable = torch.compile(model)

    device_inputs = {key: value.to(device.value) for key, value in inputs.items()}
    try:
        with torch.inference_mode(), _autocast(device, config.precision):
            output = runnable(**device_inputs)
    finally:
        for handle in handles:
            handle.remove()

    store["last_hidden_state"] = extract_logits(output).detach().to("cpu", torch.float32)
    return store


def _unhooked_output(
    model: nn.Module, inputs: dict[str, torch.Tensor], config: ExecConfig
) -> torch.Tensor:
    """Run one forward pass with no hooks and return the final hidden state.

    This is the faithful probe of a compiled configuration: with nothing materialising
    intermediate activations, the whole-graph fusion is intact and the reduced-precision
    substitution, where it occurs, is present in the output.
    """
    device = config.device
    model = model.to(device.value)
    runnable: nn.Module = model
    if config.mode is Mode.COMPILE:
        torch._dynamo.reset()
        runnable = torch.compile(model)
    device_inputs = {key: value.to(device.value) for key, value in inputs.items()}
    with torch.inference_mode(), _autocast(device, config.precision):
        output = runnable(**device_inputs)
    return extract_logits(output).detach().to("cpu", torch.float32)


def _point_metrics(baseline: torch.Tensor, target: torch.Tensor) -> LayerDivergence:
    """Compute the four divergence metrics for one capture point in fp32."""
    if baseline.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: baseline {tuple(baseline.shape)} vs {tuple(target.shape)}"
        )
    base = baseline.float()
    cand = target.float()

    diff = (base - cand).abs()
    max_abs_diff = float(diff.max())
    mean_abs_diff = float(diff.mean())

    base2d = base.reshape(-1, base.shape[-1])
    cand2d = cand.reshape(-1, cand.shape[-1])
    cosine = torch.nn.functional.cosine_similarity(base2d, cand2d, dim=1)
    cosine_similarity = float(cosine.mean())

    base_norm = float(base.norm())
    relative_error = float((base - cand).norm()) / base_norm if base_norm > 0 else 0.0

    return LayerDivergence(
        point="",
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        cosine_similarity=cosine_similarity,
        relative_error=relative_error,
    )


def compare_activations(
    baseline: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
) -> list[LayerDivergence]:
    """Compute per-capture-point divergence metrics.

    The two activation tensors of each point are released as soon as their metrics are
    computed, so the full baseline and target sets are not both held for every point.
    """
    results: list[LayerDivergence] = []
    for point in CAPTURE_POINTS:
        if point not in baseline or point not in target:
            continue
        base = baseline.pop(point)
        cand = target.pop(point)
        metrics = _point_metrics(base, cand)
        results.append(
            LayerDivergence(
                point=point,
                max_abs_diff=metrics.max_abs_diff,
                mean_abs_diff=metrics.mean_abs_diff,
                cosine_similarity=metrics.cosine_similarity,
                relative_error=metrics.relative_error,
            )
        )
        del base, cand
    return results


def first_divergence(layers: list[LayerDivergence], threshold: float) -> str | None:
    """Return the earliest capture point whose relative error exceeds ``threshold``."""
    for layer in layers:
        if layer.relative_error > threshold:
            return layer.point
    return None


def localize_run(
    config: ExecConfig,
    model_factory,
    input_factory,
    *,
    model_key: str = "distilbert",
    threshold: float = DEFAULT_THRESHOLD,
) -> LocalizationReport:
    """Localise divergence of a configuration against the fp32 eager baseline.

    ``model_factory`` returns a fresh eval-mode model on each call. A fresh model is
    used for every pass so that compilation state is never shared across the baseline
    capture, the target capture, and the un-hooked probe.
    """
    baseline_config = ExecConfig(config.device, BASELINE_PRECISION, BASELINE_MODE)
    inputs = input_factory()

    baseline = capture_activations(model_factory(), inputs, baseline_config)
    target = capture_activations(model_factory(), inputs, config)

    probe_effect: dict[str, float] | None = None
    if config.mode is Mode.COMPILE:
        baseline_final = baseline["last_hidden_state"]
        unhooked_final = _unhooked_output(model_factory(), inputs, config)
        hooked_final = target["last_hidden_state"]
        probe_effect = {
            "hooked_max_abs_diff": float((baseline_final - hooked_final).abs().max()),
            "unhooked_max_abs_diff": float((baseline_final - unhooked_final).abs().max()),
        }

    layers = compare_activations(baseline, target)
    return LocalizationReport(
        model=model_key,
        config=config.name,
        device=config.device.value,
        threshold=threshold,
        layers=layers,
        first_divergence=first_divergence(layers, threshold),
        probe_effect=probe_effect,
    )


def render_markdown(report: LocalizationReport) -> str:
    """Render a divergence localisation report in Markdown."""
    lines = ["# Divergence Localisation Report", ""]
    lines.append(
        f"Model **{report.model}**, configuration **{report.config}** on "
        f"{report.device}, compared against the fp32 eager baseline."
    )
    lines.append("")
    lines.append(
        f"Relative error is the Frobenius norm of the difference normalised by the "
        f"baseline activation norm. First divergence is the earliest capture point whose "
        f"relative error exceeds the threshold of {report.threshold:.1e}."
    )
    lines.append("")

    if report.first_divergence is None:
        lines.append("No capture point exceeds the divergence threshold.")
    else:
        lines.append(f"First divergence: **{report.first_divergence}**.")
    lines.append("")

    header = ["Capture point", "Max abs diff", "Mean abs diff", "Cosine", "Relative error"]
    rows = []
    for layer in report.layers:
        rows.append(
            [
                layer.point,
                f"{layer.max_abs_diff:.3e}",
                f"{layer.mean_abs_diff:.3e}",
                f"{layer.cosine_similarity:.5f}",
                f"{layer.relative_error:.3e}",
            ]
        )
    lines.append(markdown_table(header, rows))

    if report.probe_effect is not None:
        hooked = report.probe_effect["hooked_max_abs_diff"]
        unhooked = report.probe_effect["unhooked_max_abs_diff"]
        lines.append("")
        lines.append("## Probe effect")
        lines.append("")
        lines.append(
            "This is a compiled configuration. Forward hooks force a graph break that "
            "changes the compiled numerics, so the per-layer curve above reflects the "
            "hooked path rather than the production compiled path. The final hidden state "
            "diverges from the baseline by:"
        )
        lines.append("")
        lines.append(f"- {hooked:.3e} with hooks attached")
        lines.append(f"- {unhooked:.3e} with no hooks attached")
        lines.append("")
        if unhooked > hooked * 10:
            lines.append(
                "The un-hooked divergence is far larger: the divergence is a whole-graph "
                "property that probing dissolves, so it cannot be attributed to a single "
                "capture point."
            )
        else:
            lines.append(
                "The two agree, so probing does not alter the compiled numerics for this "
                "configuration and the per-layer curve is faithful."
            )
    return "\n".join(lines) + "\n"
