"""Compatibility name for normalized 1D frequency referencing."""

from .interop_reference import reference_interop_1d


def reference_agilent_1d(dataset, *, reference, units="ppm"):
    """Compatibility name for :func:`reference_interop_1d` (experimental API)."""
    return reference_interop_1d(dataset, reference=reference, units=units)
