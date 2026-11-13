"""In-memory equivalents of selected TopSpin baseline and slice commands."""

from operator import index

import numpy as np

_PAIRED_SPECTRA = {("S_R", "S_I"), ("F1_R", "F1_I")}


def _plane_layout(dataset, command):
    dims = tuple(dataset.dims)
    if tuple(dataset.meta.get("axis_roles") or ())[-2:] == ("F2", "F1"):
        raise ValueError(f"{command} requires F1 on y and F2 on x")
    if dims == ("y", "x"):
        return False
    if dims == ("c", "y", "x") and dataset.shape[0] in (2, 4):
        return True
    raise ValueError(f"{command} requires a 2D F1 × F2 dataset in (y, x) order")


def _real_plane_index(dataset, command):
    labels = tuple(dataset.meta.get("component_labels") or ())
    if labels in _PAIRED_SPECTRA:
        return 0
    for name in ("rr", "2rr"):
        if name in labels and not dataset.is_complex:
            return labels.index(name)
    raise ValueError(f"{command} requires identifiable real-real spectral data")


def _processed_slice(dataset, paired, command, number):
    along_f2 = command == "rsr"
    dim = "y" if along_f2 else "x"
    keys = (number, slice(None)) if along_f2 else (slice(None), number)
    complex_axis = dataset.meta.get("interop_complex_axis") or "F2"
    # This Bruker profile packs 2rr + i*2ir in its first channel.
    if dataset.meta.get("normalization_profile") == "bruker-pdata-2d-public-v1":
        complex_axis = "F2"
    retained = "F2" if along_f2 else "F1"
    if not paired:
        trace = dataset[keys].squeeze(dim)
        return trace if complex_axis == retained else trace.real

    rr = _real_plane_index(dataset, command)
    labels = tuple(dataset.meta.component_labels)
    trace = dataset[rr, *keys].squeeze("c", dim)
    if labels in _PAIRED_SPECTRA:
        if complex_axis == retained:
            return trace
        imag = dataset[1, *keys].squeeze("c", dim).real
    else:
        imag_label = "ri" if along_f2 else "ir"
        if "2rr" in labels:
            imag_label = "2ir" if along_f2 else "2ri"
        if imag_label not in labels:
            return trace
        imag = dataset[labels.index(imag_label), *keys].squeeze("c", dim)
    return trace.real + 1j * imag


def _one_based(value, count, kind):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{kind} number must be an integer")
    value = index(value)
    if not 1 <= value <= count:
        raise IndexError(f"{kind} number must be between 1 and {count}")
    return value - 1


def select_slice(dataset, domains, command, number):
    paired = _plane_layout(dataset, command)
    f1, f2 = domains[-2:]
    if command == "rser":
        if (f1, f2) != ("time", "time"):
            raise ValueError("rser requires an untransformed 2D acquisition")
        if paired and dataset.shape[0] != 2:
            raise ValueError("rser requires a pair of stored acquisition branches")
        row = _one_based(number, dataset.shape[-2] * (2 if paired else 1), "FID")
        if paired:
            result = dataset[row % 2, row // 2, :].squeeze("c", "y")
        else:
            result = dataset[row, :].squeeze("y")
        retained, frequency = "F2", False
    elif command == "rsr":
        if f2 != "frequency":
            raise ValueError("rsr requires frequency-domain F2 data")
        row = _one_based(number, dataset.shape[-2], "Row")
        if paired and f1 == "time":
            result = dataset[0, row, :].squeeze("c", "y")
        else:
            result = _processed_slice(dataset, paired, command, row)
        retained, frequency = "F2", True
    else:
        if f2 != "frequency" or f1 not in {"time", "frequency"}:
            raise ValueError("rsc requires F2-transformed data with a known F1 domain")
        column = _one_based(number, dataset.shape[-1], "Column")
        if paired and f1 == "time":
            if dataset.shape[0] != 2:
                raise ValueError("rsc requires two F1 acquisition branches after XF2")
            real = dataset[0, :, column].squeeze("c", "x").real
            imag = dataset[1, :, column].squeeze("c", "x").real
            result = real + 1j * imag
        else:
            result = _processed_slice(dataset, paired, command, column)
        retained, frequency = "F1", f1 == "frequency"

    direct_fft = (dataset.meta.get("fft_presentation") or {}).get("x")
    readonly = result.meta.readonly
    result.meta.readonly = False
    # A slice is a derived trace. The source's normalized 2D contract
    # cannot describe it, but its provenance and vendor metadata still can.
    for key in (
        "component_axis",
        "component_labels",
        "stage",
        "quadrature_mode",
        "interop_dimensions",
        "interop_physical_dimensions",
        "interop_domains",
        "interop_encoding",
        "interop_axes",
        "interop_complex_axis",
        "branch_order",
        "fft_presentation",
        "fft_presentation_transformed",
        "mqmas_xf2",
    ):
        if key in result.meta:
            result.meta.data.pop(key)
    if command == "rser" and direct_fft is not None:
        result.meta.fft_presentation = {"x": direct_fft}
    axis = -1 if retained == "F2" else -2
    for key in (
        "bf1",
        "encoding",
        "exptc",
        "nuc1",
        "physical_pivots",
        "phased",
        "phc0",
        "phc1",
        "pivot",
        "sf",
        "sfo1",
        "sw",
        "sw_h",
    ):
        values = result.meta.get(key)
        if isinstance(values, (tuple, list)) and len(values) >= 2:
            result.meta[key] = [values[axis]]
    result.meta.ndim = 1
    result.meta.datatype = "1D" if frequency else "FID"
    result.meta.td = [result.size]
    result.meta.si = [result.size]
    result.meta.iscomplex = [result.is_complex]
    result.meta.isfreq = [frequency]
    result.meta.data_domain = "frequency" if frequency else "time"
    result.meta.axis_roles = (retained,)
    result.meta.readonly = readonly
    result.history = f"Extracted {retained} trace with {command}({number})"
    return result


def _baseline_residual(section, coordinates, order, ranges):
    """Fit each trace on valid samples, with automatic or explicit support."""
    span = np.ptp(coordinates)
    if not np.isfinite(coordinates).all() or (order > 0 and span == 0):
        raise ValueError("Polynomial baseline requires finite, distinct coordinates")
    grid = (
        2 * (coordinates - coordinates.min()) / span - 1
        if span
        else np.zeros_like(coordinates)
    )
    design = np.polynomial.legendre.legvander(grid, order)
    corrected = section.data.astype(float, copy=True)
    masks = np.broadcast_to(section.mask, section.shape)
    selected = np.ones(coordinates.size, dtype=bool)
    if ranges is not None:
        selected[:] = False
        for pair in ranges:
            lower, upper = sorted(map(float, pair))
            selected |= (coordinates >= lower) & (coordinates <= upper)
    for trace, masked in zip(corrected, masks, strict=True):
        valid = ~masked & np.isfinite(trace)
        if not valid.any():
            continue
        support = valid & selected
        for iteration in range(6 if ranges is None else 1):
            coeff, _, rank, _ = np.linalg.lstsq(
                design[support], trace[support], rcond=None
            )
            if rank < order + 1:
                if iteration == 0:
                    raise ValueError(
                        "Insufficient unmasked points for the baseline polynomial"
                    )
                break
            residual = trace[valid] - design[valid] @ coeff
            if ranges is not None:
                break
            center = np.median(residual)
            sigma = 1.4826 * np.median(np.abs(residual - center))
            floor = 32 * np.finfo(float).eps * np.max(np.abs(trace[valid]))
            keep = valid.copy()
            keep[valid] = np.abs(residual - center) <= max(3 * sigma, floor)
            if keep.sum() < order + 1 or np.array_equal(keep, support):
                break
            support = keep
        trace[valid] = residual
    return corrected


def correct_baseline(dataset, domains, dim, *, order, limits, ranges):
    command = "abs2" if dim == "x" else "abs1"
    paired = _plane_layout(dataset, command)
    if tuple(domains[-2:]) != ("frequency", "frequency"):
        raise ValueError(f"{command} requires a processed 2D spectrum")
    order = index(order)
    if not 0 <= order <= 5:
        raise ValueError("Baseline polynomial order must be between 0 and 5")

    rr_index = _real_plane_index(dataset, command) if paired else None
    rr = dataset[rr_index, :, :].squeeze("c").real if paired else dataset.real
    oriented = rr if dim == "x" else rr.T
    coord = oriented.coord(dim)
    if coord is None or coord.data is None:
        raise ValueError("Polynomial baseline requires a numerical coordinate")
    coordinates = coord.data
    region = slice(None)
    if limits is not None:
        steps = np.diff(coordinates)
        if not (np.all(steps > 0) or np.all(steps < 0)):
            raise ValueError(
                "Baseline correction limits require a strictly monotonic axis"
            )
        lower, upper = sorted(float(value) for value in limits)
        selected = np.flatnonzero((coordinates >= lower) & (coordinates <= upper))
        if not selected.size:
            raise ValueError("Baseline correction limits contain no samples")
        region = slice(selected[0], selected[-1] + 1)
    corrected = _baseline_residual(
        oriented[:, region], coordinates[region], order, ranges
    )
    result = dataset.copy()
    # Promote integer spectra before subtraction and preserve other components.
    data = dataset.data.astype(
        np.result_type(dataset.dtype, corrected.dtype), copy=True
    )
    target = data[rr_index].real if paired else data.real
    if dim == "y":
        target = target.T
    target[:, region] = corrected
    result.data = data
    result.history = f"Applied {command} polynomial baseline correction (order={order})"
    return result
