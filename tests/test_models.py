"""Tests for model utilities, focused on deterministic seeding."""

from __future__ import annotations

import pytest
import torch

from mlrw.models import set_determinism

mps_available = torch.backends.mps.is_available()


def test_set_determinism_is_reproducible_on_cpu():
    set_determinism(1234)
    a = torch.randn(8)
    set_determinism(1234)
    b = torch.randn(8)
    assert torch.equal(a, b)


@pytest.mark.skipif(not mps_available, reason="MPS backend not available on this host")
def test_set_determinism_seeds_mps_generator():
    # On-device MPS random draws must reproduce after reseeding, which requires the
    # MPS generator to be seeded and not only the CPU generator.
    set_determinism(1234)
    a = torch.randn(8, device="mps")
    set_determinism(1234)
    b = torch.randn(8, device="mps")
    assert torch.equal(a, b)
