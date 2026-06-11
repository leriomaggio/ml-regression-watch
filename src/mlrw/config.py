"""Execution configuration model.

An execution configuration is the combination of a device, a numerical precision,
and an execution mode (eager or compiled). The configuration matrix is built from
these orthogonal axes, which makes adding a new device or precision a matter of
extending an enum rather than touching the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Device(StrEnum):
    """Compute device on which a model is executed."""

    CPU = "cpu"
    CUDA = "cuda"


class Precision(StrEnum):
    """Numerical precision used during the timed forward passes."""

    FP32 = "fp32"
    BF16 = "bf16"


class Mode(StrEnum):
    """Execution mode applied to the model before timing."""

    EAGER = "eager"
    COMPILE = "compile"


# Default precisions and modes that make up the configuration matrix. The baseline
# (eager, fp32) is always included because validation and regression checks are
# defined relative to it.
DEFAULT_PRECISIONS: tuple[Precision, ...] = (Precision.FP32, Precision.BF16)
DEFAULT_MODES: tuple[Mode, ...] = (Mode.EAGER, Mode.COMPILE)

# Reference configuration against which every other configuration is validated.
BASELINE_PRECISION = Precision.FP32
BASELINE_MODE = Mode.EAGER


@dataclass(frozen=True)
class ExecConfig:
    """A single execution configuration: device x precision x mode."""

    device: Device
    precision: Precision
    mode: Mode

    @property
    def name(self) -> str:
        """Stable identifier such as ``eager_fp32`` used in artifacts and reports."""
        return f"{self.mode.value}_{self.precision.value}"

    @property
    def is_baseline(self) -> bool:
        """Whether this configuration is the fp32 eager reference."""
        return self.precision is BASELINE_PRECISION and self.mode is BASELINE_MODE

    def __str__(self) -> str:
        return f"{self.device.value}:{self.name}"


def default_configs(
    device: Device,
    precisions: tuple[Precision, ...] = DEFAULT_PRECISIONS,
    modes: tuple[Mode, ...] = DEFAULT_MODES,
) -> list[ExecConfig]:
    """Build the configuration matrix for a device.

    The baseline configuration is placed first so that downstream code can rely on
    its position when establishing the reference outputs.
    """
    configs: list[ExecConfig] = []
    for mode in modes:
        for precision in precisions:
            configs.append(ExecConfig(device=device, precision=precision, mode=mode))
    configs.sort(key=lambda c: (not c.is_baseline, c.mode.value, c.precision.value))
    return configs


def resolve_device(requested: str) -> Device:
    """Resolve a requested device string, honouring automatic selection.

    ``auto`` selects CUDA when a CUDA device is visible to PyTorch and CPU otherwise.
    The import of torch is deferred so that pure-logic consumers (for example the
    test suite) do not pay the import cost.
    """
    requested = requested.lower()
    if requested == "auto":
        import torch

        return Device.CUDA if torch.cuda.is_available() else Device.CPU
    return Device(requested)
