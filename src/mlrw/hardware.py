"""Hardware and library introspection.

The information collected here is embedded in every run artifact so that results can
be interpreted in the context of the machine and software stack that produced them.
This is essential for reproducibility: a latency number is only meaningful alongside
the hardware and library versions it was measured with.
"""

from __future__ import annotations

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version


def _safe_version(package: str) -> str | None:
    """Return an installed package version, or ``None`` when it is absent."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def library_versions() -> dict[str, str | None]:
    """Collect versions of the libraries that influence numerical results."""
    return {
        "python": platform.python_version(),
        "torch": _safe_version("torch"),
        "torchvision": _safe_version("torchvision"),
        "transformers": _safe_version("transformers"),
        "numpy": _safe_version("numpy"),
        "scipy": _safe_version("scipy"),
    }


def hardware_info() -> dict[str, object]:
    """Collect a compact description of the host hardware."""
    info: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": _cpu_count(),
    }
    info.update(_cuda_info())
    return info


def _cpu_count() -> int | None:
    import os

    return os.cpu_count()


def _cuda_info() -> dict[str, object]:
    """Describe the visible CUDA device, if any."""
    try:
        import torch
    except ImportError:
        return {"cuda_available": False}

    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": ".".join(str(c) for c in torch.cuda.get_device_capability(0)),
        "cuda_version": torch.version.cuda,
    }


def git_sha() -> str | None:
    """Return the current git commit SHA, or ``None`` outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    sha = result.stdout.strip()
    return sha or None
