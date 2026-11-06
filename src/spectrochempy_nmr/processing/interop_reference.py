"""Explicit frequency referencing of normalized complex 1D acquisitions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _number(mapping, key, *, positive=False, required=True):
    value = mapping.get(key)
    if value is None and not required:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"Interop reference requires a finite numeric {key}")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"invalid interop {key}") from exc
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"Interop reference requires a finite valid {key}")
    return value


def reference_interop_1d(dataset, *, reference, units="ppm"):
    """Reference a complete normalized complex 1D FFT using producer evidence.

    ``reference`` selects an opaque key in ``acquisition.frequency_axis.offsets_hz``.
    The producer declares the sign relative to a centered negative-exponent DFT,
    the frequency of chemical zero in MHz, and any candidate offsets in Hz.
    No origin, normalization profile name, or raw vendor field is interpreted.
    No candidate is selected automatically. Cropped grids are rejected.
    """
    import spectrochempy as scp  # noqa: PLC0415

    if units not in ("ppm", "Hz"):
        raise ValueError("reference units must be 'ppm' or 'Hz'")
    if dataset.ndim != 1 or dataset.size < 2 or not np.iscomplexobj(dataset.data):
        raise ValueError("Interop reference requires one complex 1D spectrum")
    if dataset.meta.get("frequency_reference"):
        raise ValueError("this spectrum is already referenced")
    axis = dataset.coordset[dataset.dims[0]]
    if axis.units != scp.ur.Hz or tuple(dataset.meta.get("isfreq", ())) != (True,):
        raise ValueError("apply FFT first; an unreferenced Hz spectrum is required")
    acquisition = dataset.meta.get("acquisition")
    if not isinstance(acquisition, Mapping):
        raise ValueError("missing interop acquisition metadata")
    descriptor = acquisition.get("frequency_axis")
    if (
        not isinstance(descriptor, Mapping)
        or descriptor.get("schema") != "normalized-frequency-axis-v1"
    ):
        raise ValueError("missing normalized frequency-axis descriptor")
    if descriptor.get("reference_enabled") is False:
        raise ValueError("referencing is disabled")
    sign = descriptor.get("bin_sign")
    if isinstance(sign, (bool, np.bool_)) or sign not in (-1, 1):
        raise ValueError("frequency-axis bin_sign must be -1 or 1")
    offsets = descriptor.get("offsets_hz")
    if (
        not isinstance(offsets, Mapping)
        or not isinstance(reference, str)
        or reference not in offsets
    ):
        raise ValueError("choose an explicit available reference")
    sw = _number(acquisition, "spectral_width_hz", positive=True)
    frequency = _number(descriptor, "reference_frequency_mhz", positive=True)
    candidates = {key: _number(offsets, key) for key in offsets}
    center = candidates[reference]
    size = dataset.shape[0]
    step = sw / size
    if not np.isfinite(step) or step <= 0:
        raise ValueError("unrepresentable FFT frequency spacing")
    bins = (np.arange(size, dtype=float) - size // 2) * step
    # Hz conversion and CoordSet assignment can each round a coordinate view:
    # four significant digits of spacing, then three display decimals.
    rounding = 0.0005 + 0.5 * 10 ** (np.floor(np.log10(step)) - 3)
    tolerance = min(step / 4, 2 * rounding + 1e-8)
    coordinates = np.asarray(axis.data)
    if (
        coordinates.shape != (size,)
        or coordinates.dtype.kind not in "iuf"
        or not np.isfinite(coordinates).all()
        or not np.all(np.diff(coordinates) > 0)
        or not np.allclose(coordinates, bins, rtol=1e-12, atol=tolerance)
    ):
        raise ValueError("reference requires the complete ascending centered FFT grid")
    values = center + sign * bins
    if not np.isfinite(values).all():
        raise ValueError("nonfinite referenced frequency axis")
    coord = scp.Coord(values, units="Hz", rounding=False, sigdigits=15)
    coord.meta["acquisition_frequency"] = frequency * scp.ur.MHz
    if units == "ppm":
        coord.ito("ppm")
    coord.title = "chemical shift" if units == "ppm" else "referenced frequency"
    disagreement = (
        max(candidates.values()) - min(candidates.values())
        if len(candidates) > 1
        else None
    )
    result = dataset.copy()
    readonly = result.meta.readonly
    result.meta.readonly = False
    result.set_coordset(coord)
    output = result.coordset[result.dims[0]].data
    if not np.isfinite(output).all() or not np.all(sign * np.diff(output) > 0):
        raise ValueError(
            "referenced axis cannot be represented at coordinate precision"
        )
    result.meta.data_domain = "frequency"
    result.meta.frequency_reference = {
        "profile": "normalized-1d-explicit-reference-v1",
        "reference": reference,
        "reference_frequency_mhz": frequency,
        "center_hz": center,
        "offsets_hz": candidates,
        "disagreement_hz": disagreement,
        "consistent_within_0_1_hz": None
        if disagreement is None
        else disagreement <= 0.1,
        "fft_frequency_sign": sign,
        "fft_size": size,
        "bin_width_hz": step,
        "input_grid_tolerance_hz": tolerance,
        "input_grid_error_hz": float(np.max(np.abs(coordinates - bins))),
        "units": units,
    }
    result.history = f"Explicit interop 1D reference: {reference}, output {units}"
    result.meta.readonly = readonly
    return result
