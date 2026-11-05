"""Validated mathematical quadrature reconstructions."""

from __future__ import annotations

import numpy as np


STATES_TPPI = "states-tppi"
STATES_TPPI_F1COEF = (1, 0, 0, 0, 0, 0, -1, 0)
STATES_TPPI_F2_CONVENTION = "centered-inverse-unscaled"


def validate_quadrature_contract(meta, *, profile):
    """Validate the normalized quadrature conventions used by a profile."""
    if profile != STATES_TPPI:
        raise ValueError(f"unsupported quadrature reconstruction: {profile}")
    required = {
        "quadrature_mode": STATES_TPPI,
        "branch_order": ("E1", "E2"),
        "f1coef": STATES_TPPI_F1COEF,
        "tppi_alternation_state": "encoded",
        "normalization_state": "normalized",
        "f2_convention": STATES_TPPI_F2_CONVENTION,
    }
    for key, expected in required.items():
        value = meta.get(key) if hasattr(meta, "get") else None
        if key in {"branch_order", "f1coef"}:
            value = tuple(value) if value is not None else None
        if value != expected:
            raise ValueError(f"incompatible normalized quadrature metadata: {key}")


def reconstruct_f1_interferograms(data, *, profile, meta=None):
    """Reconstruct ``C_R`` and ``C_I`` for one validated profile.

    ``data`` contains the two complex branches after the direct-dimension
    transform.  This function deliberately returns arrays only; dataset
    construction and Fourier orchestration belong to the common engine.
    """
    if meta is not None:
        validate_quadrature_contract(meta, profile=profile)
    elif profile != STATES_TPPI:
        raise ValueError(f"unsupported quadrature reconstruction: {profile}")
    data = np.asarray(data)
    if data.ndim != 3 or data.shape[0] != 2 or not np.iscomplexobj(data):
        raise ValueError("quadrature reconstruction requires two complex channels")
    c_r = data[0].real - 1j * data[1].real
    c_i = data[0].imag - 1j * data[1].imag
    return np.stack([c_r, c_i], axis=0)
