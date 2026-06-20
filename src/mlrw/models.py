"""Reference models and deterministic inputs.

Two reference models are provided: a torchvision ResNet-18 (vision) and a Hugging
Face DistilBERT (text). Inputs are generated from fixed seeds so that repeated runs
produce identical tensors, which is a precondition for meaningful numerical
comparison across configurations.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# Default batch sizes per model. They are deliberately small to keep CPU runs fast
# while remaining representative of the model's compute profile.
RESNET_BATCH = 8
DISTILBERT_BATCH = 8
DISTILBERT_SEQ_LEN = 64

_DEFAULT_SEED = 1234


def set_determinism(seed: int = _DEFAULT_SEED) -> None:
    """Seed all relevant RNGs and request deterministic kernels where available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_resnet18() -> torch.nn.Module:
    """Load a randomly initialised ResNet-18 in evaluation mode.

    Random initialisation avoids a network download for the weights while keeping
    the full compute graph. Determinism comes from the fixed seed, not from the
    pretrained checkpoint.
    """
    from torchvision.models import resnet18

    set_determinism()
    model = resnet18(weights=None)
    model.eval()
    return model


def make_resnet_input(batch: int = RESNET_BATCH) -> dict[str, torch.Tensor]:
    """Create a fixed pseudo-image batch of shape ``(batch, 3, 224, 224)``."""
    set_determinism()
    pixel_values = torch.randn(batch, 3, 224, 224)
    return {"x": pixel_values}


def load_distilbert() -> torch.nn.Module:
    """Load DistilBERT in evaluation mode.

    The tokenizer is not required for inference because inputs are produced directly
    as token id tensors, so only the model weights are downloaded.
    """
    from transformers import AutoModel

    set_determinism()
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    return model


def make_distilbert_input(
    batch: int = DISTILBERT_BATCH, seq_len: int = DISTILBERT_SEQ_LEN
) -> dict[str, torch.Tensor]:
    """Create a fixed tokenized batch of input ids and an attention mask."""
    set_determinism()
    vocab_size = 30522  # distilbert-base-uncased vocabulary size
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def extract_logits(output: Any) -> torch.Tensor:
    """Reduce a model output to a single tensor used for numerical comparison.

    ResNet returns a tensor directly. Transformer models return a structured output;
    the last hidden state is used as the comparison tensor.
    """
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"Cannot extract a comparison tensor from output of type {type(output)!r}")


@dataclass(frozen=True)
class ModelSpec:
    """Description of a benchmarkable model.

    ``loader`` returns an eval-mode module and ``input_factory`` returns a mapping of
    keyword arguments passed to the module's forward method.
    """

    key: str
    task: str
    loader: Callable[[], torch.nn.Module]
    input_factory: Callable[[], dict[str, torch.Tensor]]


def default_model_specs() -> list[ModelSpec]:
    """Return the reference model specifications."""
    return [
        ModelSpec(
            key="resnet18",
            task="vision",
            loader=load_resnet18,
            input_factory=make_resnet_input,
        ),
        ModelSpec(
            key="distilbert",
            task="text",
            loader=load_distilbert,
            input_factory=make_distilbert_input,
        ),
    ]


def get_model_spec(key: str) -> ModelSpec:
    """Look up a model specification by key."""
    for spec in default_model_specs():
        if spec.key == key:
            return spec
    available = ", ".join(spec.key for spec in default_model_specs())
    raise KeyError(f"Unknown model {key!r}. Available models: {available}")
