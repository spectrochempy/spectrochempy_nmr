"""Conversion from the neutral normalized acquisition contract."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


_OPERATION_KEYS = {
    "name",
    "stage",
    "input_dimensions",
    "output_dimensions",
    "axis",
    "scale",
    "conjugation",
    "sign",
    "permutation",
    "evidence",
}


def _plain(value):
    if isinstance(value, bytes):
        return np.frombuffer(value, dtype=np.uint8).copy()
    """Convert neutral read-only mappings/diagnostics to copyable metadata."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if all(
        hasattr(value, field) for field in ("code", "severity", "message", "context")
    ):
        return {
            "code": value.code,
            "severity": value.severity,
            "message": value.message,
            "context": _plain(value.context),
        }
    return value


def _axis_coordinate(axis, size, dim, **coord_options):
    import spectrochempy as scp  # noqa: PLC0415

    if axis.domain != "time":
        raise ValueError("from_interop 1D axis must be in the time domain")
    if (
        axis.unit is not None
        and scp.ur.Unit(axis.unit).dimensionality != scp.ur.s.dimensionality
    ):
        raise ValueError("from_interop time axis units must describe time")
    if axis.values is None:
        return scp.Coord.arange(size, dims=dim, title=axis.label)
    if np.ma.is_masked(axis.values):
        raise ValueError("from_interop time axis must be unmasked")
    values = np.asarray(axis.values)
    if (
        values.shape != (size,)
        or values.dtype.kind not in "iuf"
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "from_interop time axis must be a finite real 1D array matching data"
        )
    # Coord's display rounding can collapse short dwell intervals when the
    # acquisition spans seconds. Keep the physical time values accessible.
    # The core can reconstruct Coord during assignment, so also retain a high
    # significant-digit setting if it restores display rounding on that copy.
    kwargs = {"title": axis.label, "rounding": False, "sigdigits": 15}
    if axis.unit is not None:
        kwargs["units"] = axis.unit
    return scp.Coord(np.array(values, copy=True), **kwargs, **coord_options)


def _frequency_coordinate(axis, size, **coord_options):
    """Preserve an explicitly calibrated normalized frequency grid."""
    import spectrochempy as scp  # noqa: PLC0415

    if (
        axis.domain != "frequency"
        or axis.unit not in ("Hz", "ppm")
        or axis.values is None
    ):
        raise ValueError("normalized spectrum requires explicit Hz or ppm coordinates")
    if np.ma.is_masked(axis.values):
        raise ValueError("normalized frequency axis must be unmasked")
    values = np.asarray(axis.values)
    if (
        values.shape != (size,)
        or values.dtype.kind not in "iuf"
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "normalized frequency axis must be finite and match the spectrum"
        )
    floating = values.astype(float)
    if not np.isfinite(floating).all():
        raise ValueError("normalized frequency coordinates exceed floating-point range")
    if values.dtype.kind in "iu" and not np.array_equal(
        values.astype(object), floating.astype(object)
    ):
        raise ValueError(
            "normalized integer frequency coordinates exceed floating-point precision"
        )
    values = floating
    differences = np.diff(values)
    if size < 2 or not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError("normalized frequency axis must be strictly monotonic")
    coord = scp.Coord(
        values.copy(),
        units=axis.unit,
        title=axis.label,
        rounding=False,
        sigdigits=15,
        **coord_options,
    )
    reference = getattr(axis, "reference", None)
    if reference is not None and reference.observation_frequency is not None:
        frequency = reference.observation_frequency
        if isinstance(frequency, (bool, np.bool_)) or not isinstance(
            frequency, (int, float, np.integer, np.floating)
        ):
            raise ValueError(
                "normalized reference frequency must be a positive MHz scalar"
            )
        try:
            frequency = float(frequency)
        except (ValueError, OverflowError) as exc:
            raise ValueError("invalid normalized reference frequency") from exc
        if (
            not np.isfinite(frequency)
            or frequency <= 0
            or reference.observation_frequency_unit != "MHz"
        ):
            raise ValueError(
                "normalized reference frequency must be a positive MHz scalar"
            )
        coord.meta["acquisition_frequency"] = float(frequency) * scp.ur.MHz
        coord.meta["reference_source"] = reference.source
    return coord


def _interop_metadata(result):
    if not isinstance(result.provenance, Mapping) or not isinstance(
        result.acquisition, Mapping
    ):
        raise TypeError("from_interop provenance and acquisition must be mappings")
    operations = []
    for operation in result.applied_operations:
        if not isinstance(operation, Mapping) or not _OPERATION_KEYS <= set(operation):
            raise ValueError("normalized operation does not satisfy its schema")
        operations.append(_plain(operation))
    axes = []
    for axis in result.axes:
        reference = getattr(axis, "reference", None)
        axes.append(
            {
                "label": axis.label,
                "unit": axis.unit,
                "domain": axis.domain,
                "direction": axis.direction,
                "reference": (
                    {
                        "observation_frequency": reference.observation_frequency,
                        "observation_frequency_unit": reference.observation_frequency_unit,
                        "offset": reference.offset,
                        "offset_unit": reference.offset_unit,
                        "source": reference.source,
                    }
                    if reference is not None
                    else None
                ),
            }
        )
    return {
        "normalization_profile": result.normalization_profile,
        "normalization_state": result.normalization_state,
        "quadrature_mode": result.quadrature_mode,
        "branch_order": tuple(result.branch_order),
        "interop_dimensions": tuple(result.dimensions),
        "interop_physical_dimensions": tuple(result.physical_dimensions),
        "interop_domains": tuple(result.domains),
        "interop_encoding": tuple(result.encoding),
        "provenance": _plain(result.provenance),
        "acquisition": _plain(result.acquisition),
        "applied_operations": tuple(operations),
        "diagnostics": _plain(result.diagnostics),
        "source_metadata": _plain(result.metadata),
        "interop_axes": tuple(axes),
    }


def _raw_time_coordinate(axis, size, dim):
    if axis.values is None or axis.unit != "s" or axis.direction != "increasing":
        raise ValueError(
            "from_interop raw 2D requires explicit increasing time axes in seconds"
        )
    coord = _axis_coordinate(axis, size, dim, linearize_below=0.0)
    values = coord.data
    if (
        size == 0
        or values[0] != 0
        or (
            size > 1
            and (
                values[1] <= 0
                or not np.allclose(
                    values, np.arange(size) * values[1], rtol=1e-12, atol=0
                )
            )
        )
    ):
        raise ValueError(
            "from_interop raw 2D requires uniform time coordinates starting at zero"
        )
    return coord


def _from_2d(result, data):
    """Convert a supported 2D input in its physical axis order."""
    dimensions = tuple(result.physical_dimensions)
    encoding = tuple(result.encoding)
    branches = tuple(result.branch_order)
    channel_axis = getattr(result, "channel_axis", None)
    mode = result.quadrature_mode
    domains = tuple(result.domains)
    time_qf = (
        domains == ("time", "time")
        and isinstance(result.acquisition, Mapping)
        and result.acquisition.get("indirect_sampling") == "qf"
    )
    paired_spectrum = mode == "complex-channel-pair-2d"
    if (
        dimensions not in (("F1", "F2"), ("F2", "F1"))
        or len(result.axes) != 2
        or (domains != ("frequency", "frequency") and not time_qf)
        or result.normalization_state != "storage-normalized"
        or data.dtype.kind not in ("c" if paired_spectrum else "iuf")
        or not np.isfinite(data).all()
    ):
        raise ValueError(
            "from_interop 2D requires a normalized F1/F2 frequency plane or QF FID"
        )

    stored_components = (
        mode == "stored-components-2d"
        and data.ndim == 3
        and isinstance(channel_axis, (int, np.integer))
        and not isinstance(channel_axis, bool)
        and channel_axis == 0
        and tuple(result.dimensions) == ("channels", *dimensions)
    )
    float_components = data.dtype.kind == "f" and data.dtype.itemsize in (4, 8)
    complex_axis = None
    component_labels = None
    if (
        paired_spectrum
        and not time_qf
        and data.ndim == 3
        and data.shape[0] == 2
        and isinstance(channel_axis, (int, np.integer))
        and not isinstance(channel_axis, (bool, np.bool_))
        and channel_axis == 0
        and tuple(result.dimensions) == ("channels", *dimensions)
        and encoding == ("complex", "complex")
        and branches == ("S_R", "S_I")
        and isinstance(result.acquisition, Mapping)
        and result.acquisition.get("complex_axis") in dimensions
    ):
        from .processing.fft_validation import _require_fft_prevalidation  # noqa: PLC0415

        _require_fft_prevalidation()
        component_labels = branches
        plane = np.array(data, copy=True)
    elif (
        not time_qf
        and mode == "real-2d"
        and data.ndim == 2
        and tuple(result.dimensions) == dimensions
        and encoding == ("real", "real")
        and branches == ("real",)
        and channel_axis is None
    ):
        plane = np.array(data, copy=True)
    elif (
        stored_components
        and data.shape[0] == 2
        and (
            not time_qf
            or encoding
            == tuple("complex" if dim == "F2" else "real" for dim in dimensions)
        )
        and float_components
        and (encoding, branches)
        in (
            (("real", "complex"), ("rr", "ri")),
            (("complex", "real"), ("rr", "ir")),
        )
    ):
        complex_axis = dimensions[encoding.index("complex")]
        plane = np.empty(data.shape[1:], dtype=np.result_type(data.dtype, np.complex64))
        plane.real, plane.imag = data
    elif (
        not time_qf
        and stored_components
        and data.shape[0] == 4
        and (
            (
                float_components
                and encoding == ("complex", "complex")
                and branches == ("rr", "ri", "ir", "ii")
            )
            or (
                dimensions == ("F1", "F2")
                and encoding == ("stored", "stored")
                and branches == ("2rr", "2ri", "2ir", "2ii")
                and (data.dtype.kind, data.dtype.itemsize) in (("i", 4), ("f", 8))
            )
        )
    ):
        component_labels = branches
        plane = np.array(data, copy=True)
    else:
        raise ValueError(
            "from_interop 2D requires a real plane or supported complex components"
        )

    import spectrochempy as scp  # noqa: PLC0415

    dims = ["y", "x"]
    if time_qf:
        coords = [
            _raw_time_coordinate(axis, size, dim)
            for axis, size, dim in zip(result.axes, plane.shape, dims, strict=True)
        ]
    else:
        coords = [
            _frequency_coordinate(axis, size, linearize_below=0.0)
            for axis, size in zip(result.axes, plane.shape[-2:], strict=True)
        ]
    if paired_spectrum:
        acquisition = result.acquisition.get("acquisition_frequency_mhz")
        if (
            not isinstance(acquisition, (tuple, list))
            or len(acquisition) != len(coords)
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or value <= 0
                for value in acquisition
            )
        ):
            raise ValueError(
                "normalized spectrum requires positive acquisition frequencies"
            )
        for coord, frequency in zip(coords, acquisition, strict=True):
            reference = coord.meta.get("acquisition_frequency")
            if reference is None:
                raise ValueError("normalized spectrum requires a frequency reference")
            coord.ito("ppm")
            coord.meta["acquisition_frequency"] = frequency * scp.ur.MHz
            coord.meta["reference_frequency"] = reference
    coordinates = dict(zip(dims, coords, strict=True))
    axis_roles = dimensions
    isfreq = [not time_qf, not time_qf]
    if component_labels is not None:
        dims.insert(0, "c")
        coordinates["c"] = (
            scp.Coord.arange(2, dims="c", labels=list(component_labels))
            if paired_spectrum
            else scp.Coord(labels=list(component_labels))
        )
        axis_roles = ("component", *dimensions)
        isfreq.insert(0, False)
    dataset = scp.NDDataset(plane, dtype=plane.dtype, dims=dims)
    # Named assignment retains the configured Coord instead of rebuilding it.
    dataset.set_coordset(**coordinates)
    dataset.meta.update(_interop_metadata(result))
    dataset.meta.axis_roles = axis_roles
    dataset.meta.data_domain = domains[0]
    dataset.meta.isfreq = isfreq
    if component_labels is not None:
        dataset.meta.component_axis = "c"
        dataset.meta.component_labels = component_labels
        dataset.meta.stage = "spectrum"
    dataset.meta.td = list(plane.shape)
    if paired_spectrum:
        dataset.meta.storage_profile = result.normalization_profile
        dataset.meta.interop_complex_axis = result.acquisition["complex_axis"]
    if time_qf:
        _set_ser_fft_presentation(dataset, result)
    if complex_axis is not None:
        dataset.meta.interop_complex_axis = complex_axis
    return dataset


def _set_ser_fft_presentation(dataset, result):
    import spectrochempy as scp  # noqa: PLC0415

    from .processing.fft_validation import _require_fft_prevalidation  # noqa: PLC0415

    presentation = result.acquisition.get("fft_presentation")
    if presentation is None:
        return
    _require_fft_prevalidation()
    dimensions = tuple(result.physical_dimensions)
    if not isinstance(presentation, Mapping) or set(presentation) != set(dimensions):
        raise ValueError("invalid SER FFT presentation dimensions")
    widths = result.acquisition["spectral_width_hz"]
    fft_axes = {}
    channel = dataset.ndim == 3
    indirect_transform = {
        "states-tppi": ("states-tppi-cross-pairs", "INDIRECT:STATES-TPPI"),
        "echo-antiecho-stored-2d": (
            "echo-antiecho-branch-fft",
            "INDIRECT:ECHO-ANTIECHO",
        ),
    }.get(dataset.meta.quadrature_mode)
    zero = [0 * scp.ur.degree] if channel else []
    first = zero.copy()
    encoding = ["CHANNELS"] if channel else []
    for dim, role, width in zip(dataset.dims[-2:], dimensions, widths, strict=True):
        descriptor = presentation[role]
        transform = descriptor.get("transform")
        if transform == "conjugate-input" and not (channel and role == "F1"):
            fft_encoding = "CONJUGATE-FFT"
        elif role == "F1" and indirect_transform and transform == indirect_transform[0]:
            fft_encoding = indirect_transform[1]
        else:
            raise ValueError("unsupported SER FFT presentation transform")
        phase = descriptor["initial_phase"]
        if phase.get("pivot_index") != 0:
            raise ValueError("unsupported SER FFT phase pivot")
        angles = float(phase["zero_order_deg"]), float(phase["first_order_deg"])
        if not np.isfinite(angles).all():
            raise ValueError("invalid SER FFT initial phase")
        zero.append(angles[0] * scp.ur.degree)
        first.append(angles[1] * scp.ur.degree)
        encoding.append("PENDING-F2:" + fft_encoding if role == "F1" else fft_encoding)
        fft_axes[dim] = {
            "physical_axis": role,
            "encoding": fft_encoding,
            "fft_display_axis": _plain(descriptor.get("display_axis")),
            "spectral_width_hz": width,
            "nucleus": dataset.coordset[dim].title,
        }
    dataset.meta.fft_presentation = fft_axes
    dataset.meta.encoding = encoding
    dataset.meta.phc0 = zero
    dataset.meta.phc1 = first
    dataset.meta.pivot = [0] * dataset.ndim


def _from_states_tppi(result, data):
    """Preserve E1/E2 in the explicitly declared States-TPPI profile."""
    channel_axis = getattr(result, "channel_axis", None)
    if (
        tuple(result.physical_dimensions) != ("F1", "F2")
        or tuple(result.dimensions) != ("channels", "F1", "F2")
        or tuple(result.domains) != ("time", "time")
        or tuple(result.encoding) != ("complex", "complex")
        or tuple(result.branch_order) != ("rr", "ri", "ir", "ii")
        or result.quadrature_mode != "stored-components-2d"
        or result.normalization_state != "storage-normalized"
        or not isinstance(channel_axis, (int, np.integer))
        or isinstance(channel_axis, bool)
        or channel_axis != 0
        or len(result.axes) != 2
        or data.ndim != 3
        or data.shape[0] != 4
        or data.dtype.kind != "f"
        or data.dtype.itemsize not in (4, 8)
        or not np.isfinite(data).all()
    ):
        raise ValueError(
            "States-TPPI requires normalized F1/F2 time-domain component planes"
        )

    channels = np.empty(
        (2, *data.shape[1:]), dtype=np.result_type(data.dtype, np.complex64)
    )
    channels.real = data[::2]
    channels.imag = data[1::2]
    dataset = _raw_channel_dataset(result, channels, ("E1", "E2"))
    dataset.meta.quadrature_mode = "states-tppi"
    dataset.meta.tppi_alternation_state = "encoded"
    _set_ser_fft_presentation(dataset, result)
    return dataset


def _from_echo_antiecho(result, data):
    """Preserve both complex branches in their declared storage order."""
    channel_axis = getattr(result, "channel_axis", None)
    if (
        tuple(result.physical_dimensions) != ("F1", "F2")
        or tuple(result.dimensions) != ("channels", "F1", "F2")
        or tuple(result.domains) != ("time", "time")
        or tuple(result.encoding) != ("echo-antiecho", "complex")
        or tuple(result.branch_order) != ("first", "second")
        or result.normalization_state != "storage-normalized"
        or not isinstance(channel_axis, (int, np.integer))
        or isinstance(channel_axis, bool)
        or channel_axis != 0
        or len(result.axes) != 2
        or data.ndim != 3
        or data.shape[0] != 2
        or data.dtype.kind != "c"
        or data.dtype.itemsize not in (8, 16)
        or not np.isfinite(data).all()
    ):
        raise ValueError(
            "Echo/anti-echo requires normalized F1/F2 time-domain complex branches"
        )
    dataset = _raw_channel_dataset(result, data.copy(), tuple(result.branch_order))
    _set_ser_fft_presentation(dataset, result)
    return dataset


def _from_stored_states(result, data):
    """Preserve acquired States branches with the indirect FFT guarded."""
    from .processing.fft_validation import _require_fft_prevalidation  # noqa: PLC0415

    if (
        tuple(result.physical_dimensions) != ("F1", "F2")
        or tuple(result.dimensions) != ("channels", "F1", "F2")
        or tuple(result.domains) != ("time", "time")
        or tuple(result.encoding) != ("states", "complex")
        or tuple(result.branch_order) != ("branch_0", "branch_1")
        or result.normalization_state != "storage-normalized"
        or getattr(result, "channel_axis", None) != 0
        or isinstance(result.channel_axis, (bool, np.bool_))
        or len(result.axes) != 2
        or data.ndim != 3
        or data.shape[0] != 2
        or 0 in data.shape
        or data.dtype.kind != "c"
        or data.dtype.itemsize not in (8, 16)
        or not np.isfinite(data).all()
    ):
        raise ValueError(
            "States requires normalized F1/F2 complex time-domain branches"
        )
    _require_fft_prevalidation()
    dataset = _raw_channel_dataset(
        result, data.copy(), tuple(result.branch_order), allow_unknown_time=True
    )
    dataset.meta.encoding = ["CHANNELS", "UNVALIDATED:STATES", "DIRECT:QSIM"]
    dataset.meta.iscomplex = [False, True, True]
    return dataset


def _raw_time_or_index_coordinate(axis, size, dim):
    if axis.values is None and axis.unit is None:
        return _axis_coordinate(axis, size, dim)
    return _raw_time_coordinate(axis, size, dim)


def _from_raw_trace_matrix(result, data, *, series):
    """Preserve independent complex FIDs with the leading FFT guarded."""
    import spectrochempy as scp  # noqa: PLC0415

    from .processing.fft_validation import _require_fft_prevalidation  # noqa: PLC0415

    dimensions = ("trace", "F2") if series else ("F1", "F2")
    domains = ("unknown", "time") if series else ("time", "time")
    encoding = ("unknown", "complex") if series else ("tppi", "complex")
    if (
        tuple(result.physical_dimensions) != dimensions
        or tuple(result.dimensions) != dimensions
        or tuple(result.domains) != domains
        or tuple(result.encoding) != encoding
        or tuple(result.branch_order) != ("complex",)
        or result.normalization_state != "storage-normalized"
        or getattr(result, "channel_axis", None) is not None
        or len(result.axes) != 2
        or data.ndim != 2
        or 0 in data.shape
        or data.dtype.kind != "c"
        or data.dtype.itemsize not in (8, 16)
        or not np.isfinite(data).all()
    ):
        raise ValueError("Raw trace matrix requires normalized complex F2 traces")
    legacy_series = (
        series
        and isinstance(result.acquisition, Mapping)
        and (result.acquisition.get("legacy_presentation") is not None)
    )
    if not legacy_series:
        _require_fft_prevalidation()
    leading, direct = result.axes
    if series:
        if (
            leading.domain != "unknown"
            or leading.values is not None
            or leading.unit is not None
        ):
            raise ValueError("Trace series requires an uncalibrated trace axis")
        y = scp.Coord.arange(data.shape[0], title=leading.label)
    else:
        y = _raw_time_or_index_coordinate(leading, data.shape[0], "y")
    x = _raw_time_or_index_coordinate(direct, data.shape[1], "x")
    dataset = scp.NDDataset(data.copy(), dtype=data.dtype, dims=["y", "x"])
    dataset.set_coordset(y=y, x=x)
    dataset.meta.update(_interop_metadata(result))
    dataset.meta.update(
        {
            "axis_roles": list(dimensions) if series else dimensions,
            "data_domain": "unknown" if series and not legacy_series else "time",
            "encoding": [
                "QF"
                if legacy_series
                else "UNVALIDATED:TRACE"
                if series
                else "UNVALIDATED:TPPI",
                "QSIM",
            ],
            "iscomplex": [False, True],
            "isfreq": [False, False],
            "stage": "acquisition",
            "td": list(data.shape),
        }
    )
    return dataset


def _raw_channel_dataset(result, channels, labels, *, allow_unknown_time=False):
    import spectrochempy as scp  # noqa: PLC0415

    y, x = (
        _raw_time_or_index_coordinate(axis, size, dim)
        if allow_unknown_time
        else _raw_time_coordinate(axis, size, dim)
        for axis, size, dim in zip(
            result.axes, channels.shape[1:], ("y", "x"), strict=True
        )
    )
    dataset = scp.NDDataset(channels, dtype=channels.dtype, dims=["c", "y", "x"])
    c = (
        scp.Coord.arange(2, dims="c", labels=list(labels))
        if allow_unknown_time
        else scp.Coord(labels=list(labels))
    )
    dataset.set_coordset(c=c, y=y, x=x)
    dataset.meta.update(_interop_metadata(result))
    dataset.meta.update(
        {
            "axis_roles": ("component", "F1", "F2"),
            "component_axis": "c",
            "component_labels": labels,
            "stage": "acquisition",
            "data_domain": "time",
            "isfreq": [False, False, False],
            "td": list(channels.shape),
        }
    )
    return dataset


def from_interop(result, *, profile=None):
    """Convert a normalized complex 1D FID or a supported frequency spectrum.

    The converter consumes the neutral normalized protocol by attributes and
    deliberately does not import nmrglue-ng or inspect vendor metadata. A raw
    descriptive ``ReadResult`` is rejected because it has not established the
    normalization guarantees required by this conversion. Frequency spectra
    can be real or complex in 1D. Two-dimensional spectra can be real, complex
    along one axis, or represented as four labelled real component planes.
    Raw 2D QF requires explicit provider normalization. For a known
    States-TPPI acquisition, profile="states-tppi" confirms that the
    input channels are adjacent E1/E2 pairs with alternation still encoded.
    Explicitly normalized echo/anti-echo input retains both complex branches
    in storage order. Stored States branches retain the indirect FFT guard.
    """
    required = (
        "data",
        "axes",
        "dimensions",
        "physical_dimensions",
        "domains",
        "quadrature_mode",
        "branch_order",
        "normalization_state",
        "normalization_profile",
        "applied_operations",
        "provenance",
        "acquisition",
        "metadata",
        "encoding",
        "diagnostics",
    )
    missing = [name for name in required if not hasattr(result, name)]
    if missing:
        raise TypeError(
            "from_interop requires a normalized acquisition; missing "
            + ", ".join(missing)
        )
    if np.ma.is_masked(result.data):
        raise ValueError("from_interop currently requires unmasked acquisition data")
    data = np.asarray(result.data)
    if profile == "states-tppi":
        return _from_states_tppi(result, data)
    if profile is not None:
        raise ValueError(f"unsupported interop profile: {profile!r}")
    if result.quadrature_mode == "tppi-stored-2d":
        return _from_raw_trace_matrix(result, data, series=False)
    if result.quadrature_mode == "series-stored-2d":
        return _from_raw_trace_matrix(result, data, series=True)
    if result.quadrature_mode == "states-stored-2d":
        return _from_stored_states(result, data)
    if result.quadrature_mode == "echo-antiecho-stored-2d":
        return _from_echo_antiecho(result, data)
    if data.ndim == 2 or (
        data.ndim == 3
        and result.quadrature_mode
        in (
            "stored-components-2d",
            "complex-channel-pair-2d",
        )
    ):
        return _from_2d(result, data)
    if data.ndim != 1 or data.dtype.kind not in "iufc":
        raise ValueError(
            "from_interop requires a numeric 1D acquisition or a supported 2D frequency plane"
        )
    domain = tuple(result.domains)
    if domain not in (("time",), ("frequency",)):
        raise ValueError("from_interop 1D input must declare time or frequency domain")
    complex_data = np.iscomplexobj(data)
    if domain == ("time",) and not complex_data:
        raise ValueError("from_interop time-domain input requires complex samples")
    if tuple(result.dimensions) != ("F2",) or tuple(result.physical_dimensions) != (
        "F2",
    ):
        raise ValueError("from_interop 1D input must declare the F2 dimension")
    if len(result.axes) != 1:
        raise ValueError("from_interop 1D input must have one physical axis")
    mode = "complex" if complex_data else "real"
    if result.quadrature_mode != mode + "-1d" or tuple(result.branch_order) != (mode,):
        raise ValueError("from_interop 1D input has an unsupported quadrature contract")
    if result.normalization_state not in (
        "storage-normalized",
        "quadrature-normalized",
    ):
        raise ValueError("from_interop requires a validated normalized acquisition")
    import spectrochempy as scp  # noqa: PLC0415

    axis = result.axes[0]
    dataset = scp.NDDataset(
        np.array(data, copy=True),
        coordset=[
            _axis_coordinate(axis, data.size, "x")
            if domain == ("time",)
            else _frequency_coordinate(axis, data.size)
        ],
        dims=["x"],
    )
    dataset.meta.update(_interop_metadata(result))
    dataset.meta.axis_roles = ("F2",)
    dataset.meta.data_domain = domain[0]
    dataset.meta.isfreq = [domain == ("frequency",)]
    dataset.meta.td = [int(data.size)]
    if domain == ("time",):
        presentation = result.acquisition.get("legacy_presentation")
        if presentation is not None:
            if (
                not isinstance(presentation, Mapping)
                or presentation.get("fft_input_transform") != "conjugate-input"
                or tuple(result.encoding) != ("QSIM",)
            ):
                raise ValueError("unsupported 1D interop FFT presentation")
            dataset.meta.encoding = list(result.encoding)
        phase = result.acquisition.get("fft_initial_phase")
        if phase is not None:
            if not isinstance(phase, Mapping) or phase.get("pivot_index") != 0:
                raise ValueError("unsupported interop FFT initial phase")
            zero = float(phase["zero_order_deg"])
            first = float(phase["first_order_deg"])
            if not np.isfinite([zero, first]).all():
                raise ValueError("invalid interop FFT initial phase")
            dataset.meta.pivot = [0]
            dataset.meta.phc0 = [zero * scp.ur.degree]
            dataset.meta.phc1 = [first * scp.ur.degree]
    return dataset
