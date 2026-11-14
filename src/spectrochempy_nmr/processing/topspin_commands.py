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


def extract_component(dataset, domains, component):
    """Extract a real TopSpin plane from a frequency-domain spectrum."""
    paired = _plane_layout(dataset, component.upper())
    if domains[-2:] != ("frequency", "frequency"):
        raise ValueError("Component extraction requires frequency-domain F1 and F2")

    if not paired:
        if component == "rr":
            return dataset.real
        complex_axis = dataset.meta.get("interop_complex_axis") or "F2"
        if dataset.is_complex and component == {"F1": "ri", "F2": "ir"}.get(
            complex_axis
        ):
            return dataset.imag
        raise ValueError(f"The spectrum does not contain component {component.upper()}")

    labels = tuple(dataset.meta.get("component_labels") or ())
    if not dataset.is_complex:
        # Native interop labels are in physical (F1, F2) order.
        labels = tuple(
            label[1:] if label.startswith("2") else label[::-1] for label in labels
        )
    if paired and labels in _PAIRED_SPECTRA and dataset.is_complex:
        complex_axis = dataset.meta.get("interop_complex_axis") or "F2"
        if dataset.meta.get("normalization_profile") == "bruker-pdata-2d-public-v1":
            complex_axis = "F2"
        part, branch = component if complex_axis == "F2" else component[::-1]
        plane = dataset[int(branch == "i"), :, :].squeeze("c")
        result = plane.imag if part == "i" else plane.real
    elif paired and not dataset.is_complex and component in labels:
        result = dataset[labels.index(component), :, :].squeeze("c")
    else:
        raise ValueError(f"The spectrum does not contain component {component.upper()}")

    readonly = result.meta.readonly
    result.meta.readonly = False
    for key in (
        "component_axis",
        "component_labels",
        "interop_complex_axis",
        "quadrature_mode",
        "interop_dimensions",
        "interop_physical_dimensions",
        "interop_domains",
        "interop_encoding",
        "interop_axes",
        "branch_order",
        "fft_presentation",
        "fft_presentation_transformed",
    ):
        if key in result.meta:
            result.meta.data.pop(key)
    for key in (
        "bf1",
        "exptc",
        "nuc1",
        "phased",
        "phc0",
        "phc1",
        "physical_pivots",
        "pivot",
        "sf",
        "sfo1",
        "sw",
        "sw_h",
    ):
        values = result.meta.get(key)
        if isinstance(values, (tuple, list)) and len(values) == 3:
            result.meta[key] = values[-2:]
    result.meta.ndim = 2
    result.meta.datatype = "2D"
    result.meta.isfreq = [True, True]
    result.meta.iscomplex = [False, False]
    result.meta.encoding = ["QF", "QF"]
    result.meta.axis_roles = ("F1", "F2")
    result.meta.td = list(result.shape)
    result.meta.si = list(result.shape)
    result.meta.readonly = readonly
    result.history = f"Extracted TopSpin {component.upper()} plane"
    return result


def _trace_channels(dataset, paired, along_f2, number, command, *, mixed=False):
    """Return the array indices for the real and imaginary parts of a trace."""
    keys = (number, slice(None)) if along_f2 else (slice(None), number)
    complex_axis = dataset.meta.get("interop_complex_axis") or "F2"
    if dataset.meta.get("normalization_profile") == "bruker-pdata-2d-public-v1":
        complex_axis = "F2"
    retained = "F2" if along_f2 else "F1"
    if not paired:
        return keys, keys if dataset.is_complex and complex_axis == retained else None

    if mixed:
        if not along_f2 and dataset.shape[0] != 2:
            raise ValueError(f"{command} requires two acquisition branches after XF2")
        return (0, *keys), ((0 if along_f2 else 1), *keys)

    rr = (_real_plane_index(dataset, command), *keys)
    labels = tuple(dataset.meta.component_labels)
    if labels in _PAIRED_SPECTRA:
        return rr, rr if complex_axis == retained else (1, *keys)
    imag_label = "ri" if along_f2 else "ir"
    if "2rr" in labels:
        imag_label = "2ir" if along_f2 else "2ri"
    return rr, (labels.index(imag_label), *keys) if imag_label in labels else None


def _processed_slice(dataset, paired, command, number, *, mixed=False):
    along_f2 = command == "rsr"
    dims = ("y" if along_f2 else "x",)
    if paired:
        dims = ("c", *dims)
    real, imag = _trace_channels(
        dataset, paired, along_f2, number, command, mixed=mixed
    )
    trace = dataset[real].squeeze(*dims)
    if imag == real:
        return trace
    if imag is None:
        return trace.real
    return trace.real + 1j * dataset[imag].squeeze(*dims).real


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
        result = _processed_slice(dataset, paired, command, row, mixed=f1 == "time")
        retained, frequency = "F2", True
    else:
        if f2 != "frequency" or f1 not in {"time", "frequency"}:
            raise ValueError("rsc requires F2-transformed data with a known F1 domain")
        column = _one_based(number, dataset.shape[-1], "Column")
        result = _processed_slice(dataset, paired, command, column, mixed=f1 == "time")
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


def replace_slice(dataset, domains, command, number, trace):
    """Replace the same channels exposed by the corresponding read command."""
    from spectrochempy import NDDataset

    paired = _plane_layout(dataset, command)
    f1, f2 = domains[-2:]
    along_f2 = command == "wsr"
    if f2 != "frequency" or f1 not in {"time", "frequency"}:
        raise ValueError(
            f"{command} requires F2-transformed data with a known F1 domain"
        )
    axis, dimension = (-1, "x") if along_f2 else (-2, "y")
    location = _one_based(
        number, dataset.shape[-2 if along_f2 else -1], "Row" if along_f2 else "Column"
    )
    real, imag = _trace_channels(
        dataset, paired, along_f2, location, command, mixed=f1 == "time"
    )
    if not isinstance(trace, NDDataset) or trace.shape != (dataset.shape[axis],):
        raise ValueError(f"{command} requires a 1D NDDataset of matching length")
    domain = "frequency" if along_f2 else f1
    if trace.meta.get("data_domain") not in (None, domain):
        raise ValueError("Trace and destination must have the same domain")
    source_coord, target_coord = trace.coord(0), dataset.coord(dimension)
    if (
        source_coord is None
        or target_coord is None
        or source_coord.data is None
        or target_coord.data is None
        or source_coord.units != target_coord.units
        or not np.allclose(source_coord.data, target_coord.data, rtol=1e-10, atol=0)
    ):
        raise ValueError(
            "Trace and destination must share the same coordinate grid and units"
        )
    if trace.units != dataset.units:
        raise ValueError("Trace and destination must have the same signal units")
    if imag is None and trace.is_complex:
        raise ValueError("Destination has no imaginary component for this trace")

    mask = np.broadcast_to(dataset.mask, dataset.shape).copy()
    trace_mask = np.broadcast_to(trace.mask, trace.shape)
    channels = (real,) if imag is None or imag == real else (real, imag)
    shares_mask = dataset.is_complex and imag != real
    combined_mask = mask[real] if imag is None else mask[real] | mask[imag]
    same_mask = np.array_equal(combined_mask, trace_mask)
    if shares_mask and not same_mask:
        raise ValueError("Cannot change a mask shared with an untouched component")
    if not same_mask:
        for key in channels:
            mask[key] = trace_mask

    values = trace.data if imag == real else trace.data.real
    data = dataset.data.astype(np.result_type(dataset.dtype, values.dtype), copy=True)
    if imag == real:
        data[real] = values
    else:
        writable = ~trace_mask if same_mask and imag is not None else slice(None)
        data.real[real][writable] = values[writable]
        if imag is not None:
            data.real[imag][writable] = trace.data.imag[writable]
    result = dataset.copy()
    result.data = data
    result.remove_masks()
    result.mask = mask
    result.history = f"Replaced trace with {command}({number})"
    return result


def estimate_f2_phase(trace, row, exclude, options):
    """Estimate the relative phase and identify the row on failure."""
    from .autophase import autophase

    try:
        estimate = autophase(trace, exclude=exclude, apply=False, **options).estimate
    except ValueError as exc:
        raise ValueError(f"Phase estimation failed on row {row}: {exc}") from exc
    if not estimate.success:
        raise ValueError(f"Phase estimation failed on row {row}: {estimate.message}")
    return estimate


def phase_f2_rows(dataset, read_row, reference_row, per_row, exclude, options):
    """Phase a matrix of complex F2 traces using the 1D estimator."""
    if tuple(dataset.dims) != ("y", "x") or not dataset.is_complex:
        raise ValueError("apk2d requires a matrix of complex F2 traces or MQMAS XF2")
    if dataset.meta.get("interop_complex_axis") not in (None, "F2"):
        raise ValueError("apk2d requires complex quadrature along F2")
    count, size = dataset.shape
    rows = range(1, count + 1) if per_row else (reference_row,)
    if per_row and "references" in options:
        options = {
            **options,
            "references": tuple(tuple(ref) for ref in options["references"]),
        }
    angles = []
    for row in rows:
        estimate = estimate_f2_phase(read_row(row), row, exclude, options)
        angles.append((estimate.p0, estimate.p1))

    p0, p1 = np.asarray(angles).T
    ramp = p0[:, None] + p1[:, None] * (np.arange(size) / size)
    result = dataset.copy()
    result.data = dataset.data * np.exp(1j * np.deg2rad(ramp))
    result._append_history_entry(
        operation="nmr.apk2d",
        parameters={
            "reference_row": None if per_row else reference_row,
            "per_row": per_row,
            "angles_deg": [list(pair) for pair in angles],
            "pivot_index": 0,
        },
        message="Applied automatic F2 phase correction",
    )
    return result
