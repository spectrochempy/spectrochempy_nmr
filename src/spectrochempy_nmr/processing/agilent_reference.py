"""Explicit referencing of qualified Agilent 1D interop FFT results."""

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
        raise ValueError(f"Agilent reference requires a finite numeric {key}")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"invalid Agilent {key}") from exc
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"Agilent reference requires a finite valid {key}")
    return value


def reference_agilent_1d(dataset, *, reference, units="ppm"):
    """Reference a complete Agilent interop 1D FFT, returning a copy.

    Parameters
    ----------
    dataset : NDDataset
        An unreferenced complex 1D interop FFT on its full ascending Hz grid.
    reference : {'rfl-rfp', 'sfrq-reffrq'}
        Explicit choice of carrier relative to chemical zero: ``sw/2-rfl+rfp``
        or ``(sfrq-reffrq)*1e6``. No model is selected automatically, even when
        they agree. Both use positive ``reffrq`` (MHz) for Hz/ppm conversion.
    units : {'ppm', 'Hz'}, optional
        Output coordinate units. Samples, masks and original metadata are kept.

    Notes
    -----
    Agilent's default var2pipe sign adjustment conjugates the stored FID.
    Its positive-sign DFT is the conjugate of the core's negative-sign DFT.
    Thus chemical frequency is carrier minus the core's centered FFT bins.
    This operation changes only the axis, not the complex phase convention.
    Choosing a model records conflicting source evidence; it does not establish
    which chemical reference is experimentally correct. ``tof`` is never used
    as a chemical offset. Cropped, reordered and already referenced axes are
    rejected. No core changes or external provider import are required.
    """
    import spectrochempy as scp  # noqa: PLC0415

    if reference not in ("rfl-rfp", "sfrq-reffrq"):
        raise ValueError("choose reference='rfl-rfp' or reference='sfrq-reffrq'")
    if units not in ("ppm", "Hz"):
        raise ValueError("reference units must be 'ppm' or 'Hz'")
    if dataset.ndim != 1 or dataset.size < 2 or not np.iscomplexobj(dataset.data):
        raise ValueError("Agilent reference requires one complex 1D spectrum")
    if dataset.meta.get("normalization_profile") != "agilent-fid-1d-complex-v1":
        raise ValueError("Agilent reference requires the qualified interop profile")
    if dataset.meta.get("frequency_reference"):
        raise ValueError("this spectrum is already referenced")
    axis = dataset.coordset[dataset.dims[0]]
    if axis.units != scp.ur.Hz or tuple(dataset.meta.get("isfreq", ())) != (True,):
        raise ValueError("apply FFT first; an unreferenced Hz spectrum is required")
    acquisition = dataset.meta.get("acquisition")
    if not isinstance(acquisition, Mapping):
        raise ValueError("missing Agilent acquisition metadata")
    original = acquisition.get("original_reference")
    if not isinstance(original, Mapping):
        raise ValueError("missing original Agilent reference metadata")
    refpos = dataset.meta.get("source_metadata", {}).get("procpar", {}).get("refpos")
    if refpos is not None:
        flags = refpos.get("values", ())
        if len(flags) != 1 or str(flags[0]).strip().lower() == "n":
            raise ValueError("Agilent referencing is disabled or refpos is ambiguous")
    sw = _number(acquisition, "spectral_width_hz", positive=True)
    frequency = _number(original, "reffrq", positive=True)
    obs = _number(
        acquisition, "observation_frequency_mhz", positive=True, required=False
    )
    rfl = _number(original, "rfl", required=False)
    rfp = _number(original, "rfp", required=False)
    graphical = None if rfl is None or rfp is None else sw / 2 - rfl + rfp
    absolute = None if obs is None else (obs - frequency) * 1e6
    if any(
        value is not None and not np.isfinite(value) for value in (graphical, absolute)
    ):
        raise ValueError("nonfinite Agilent carrier evidence")
    center = graphical if reference == "rfl-rfp" else absolute
    if center is None or not np.isfinite(center):
        raise ValueError(f"incomplete or nonfinite {reference} reference")
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
    values = center - bins
    if not np.isfinite(values).all():
        raise ValueError("nonfinite referenced frequency axis")
    coord = scp.Coord(values, units="Hz", rounding=False, sigdigits=15)
    coord.meta["acquisition_frequency"] = frequency * scp.ur.MHz
    if units == "ppm":
        coord.ito("ppm")
    coord.title = "chemical shift" if units == "ppm" else "referenced frequency"
    disagreement = (
        abs(graphical - absolute)
        if graphical is not None and absolute is not None
        else None
    )
    result = dataset.copy()
    readonly = result.meta.readonly
    result.meta.readonly = False
    result.set_coordset(coord)
    output = result.coordset[result.dims[0]].data
    if not np.isfinite(output).all() or not np.all(np.diff(output) < 0):
        raise ValueError(
            "referenced axis cannot be represented at coordinate precision"
        )
    result.meta.data_domain = "frequency"
    result.meta.frequency_reference = {
        "profile": "agilent-1d-explicit-reference-v1",
        "reference": reference,
        "reference_frequency_mhz": frequency,
        "center_hz": center,
        "graphical_center_hz": graphical,
        "absolute_center_hz": absolute,
        "disagreement_hz": disagreement,
        "consistent_within_0_1_hz": None
        if disagreement is None
        else disagreement <= 0.1,
        "fft_frequency_sign": -1,
        "fft_size": size,
        "bin_width_hz": step,
        "input_grid_tolerance_hz": tolerance,
        "input_grid_error_hz": float(np.max(np.abs(coordinates - bins))),
        "units": units,
    }
    result.history = f"Explicit Agilent 1D reference: {reference}, output {units}"
    result.meta.readonly = readonly
    return result
