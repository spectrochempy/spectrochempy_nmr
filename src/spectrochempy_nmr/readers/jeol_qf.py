"""Adapt an explicitly declared uniform QF JDF at the reader boundary."""

from numbers import Real

from nmrglue import interop, jeol
import numpy as np

from ..interop import from_interop


def _number(params, key, *, positive=False):
    value = params.get(key)
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or (positive and value <= 0)
    ):
        raise ValueError(f"Explicit JEOL QF requires a finite physical {key}")
    return float(value)


def read_jeol_qf(path):
    """Decode two sections; the caller, not the experiment name, declares QF."""
    # The binary provider reader supports float32 as well as float64 JDFs.
    # Its legacy complex section-pair convention is explicit at this boundary.
    metadata, data = jeol.read(path)
    header, params = metadata.get("header", {}), metadata.get("parameters", {})
    data = np.asarray(data)
    if (
        header.get("data_format") != "two_d"
        or header.get("data_type") not in ("float32", "float64")
        or tuple(header.get("data_axis_type", ())[:2])
        != ("real_complex", "real_complex")
        or tuple(header.get("data_axis_ranged", ())[:2]) != ("ranged", "ranged")
        or tuple(header.get("reversed", ())[:2]) != (False, True)
        or data.ndim != 2
        or not np.iscomplexobj(data)
        or not np.isfinite(data).all()
        or min(data.shape) < 2
    ):
        raise ValueError(
            "Explicit JEOL QF requires the qualified uniform two-section layout"
        )
    points = tuple(reversed(data.shape))
    if (
        tuple(header.get("data_points", ())[:2]) != points
        or tuple(header.get("data_offset_start", ())[:2]) != (0, 0)
        or tuple(header.get("data_offset_stop", ())[:2]) != tuple(n - 1 for n in points)
        or tuple(header.get("data_axis_start", ())[:2]) != (0, 0)
        or tuple(header.get("units", ())[:2])
        != (("none", 1, "second"), ("none", 1, "second"))
        or (params.get("x_points"), params.get("y_points")) != points
    ):
        raise ValueError("Explicit JEOL QF requires complete uniform time axes")
    widths, frequencies, offsets = [], [], []
    base = header.get("base_freq", ())
    stops = header.get("data_axis_stop", ())
    if len(base) < 2 or len(stops) < 2:
        raise ValueError("Explicit JEOL QF requires physical axis references")
    for i, axis in ((1, "y"), (0, "x")):
        width = _number(params, f"{axis}_sweep", positive=True)
        frequency = _number(params, f"{axis}_freq", positive=True) / 1e6
        offset = _number(params, f"{axis}_offset")
        if not np.isclose(base[i], frequency, rtol=1e-7, atol=0) or not np.isclose(
            stops[i], (points[i] - 1) / width, rtol=1e-7, atol=0
        ):
            raise ValueError("JEOL QF header and physical calibration disagree")
        widths.append(width)
        frequencies.append(frequency)
        offsets.append(offset * frequency)
    # The qualified direct grid places the carrier half an acquired bin below
    # its central FFT bin. The offset remains fixed when the FFT size changes.
    offsets[1] -= widths[1] / (2 * data.shape[1])
    acquisition = {
        "spectral_width_hz": tuple(widths),
        "sampling": ("regular", "regular"),
        "nominal_shape": data.shape,
        "indirect_sampling": "qf",
        "acquired_shape": data.shape,
        "acquisition_frequency_mhz": tuple(frequencies),
        "reference_frequency_mhz": tuple(frequencies),
        "frequency_offset_hz": tuple(offsets),
        "nyquist_edge": ("negative", "positive"),
        "nuclei": tuple(params.get(f"{axis}_domain") for axis in ("y", "x")),
        "quadrature_selection": {"mode": "qf", "source": "caller"},
    }
    normalized = interop.NormalizedAcquisition(
        data=np.stack((data.real, data.imag)),
        axes=tuple(
            interop.Axis(
                values=np.arange(n) / width,
                unit="s",
                label=nucleus,
                domain="time",
                direction="increasing",
            )
            for n, width, nucleus in zip(
                data.shape, widths, acquisition["nuclei"], strict=True
            )
        ),
        dimensions=("channels", "F1", "F2"),
        physical_dimensions=("F1", "F2"),
        domains=("time", "time"),
        encoding=("real", "complex"),
        quadrature_mode="stored-components-2d",
        branch_order=("rr", "ri"),
        channel_axis=0,
        normalization_state="storage-normalized",
        normalization_profile="jeol-declared-qf-2d-v1",
        acquisition=acquisition,
        provenance={"source": str(path), "format": "jeol-jdf"},
        metadata=metadata,
        applied_operations=(
            {
                "name": "decode-and-declare-qf",
                "stage": "acquisition",
                "input_dimensions": ("section", "F1", "F2"),
                "output_dimensions": ("channels", "F1", "F2"),
                "axis": "F2",
                "scale": 1.0,
                "conjugation": "conjugate",
                "sign": "none",
                "permutation": None,
                "evidence": "provider complex section-pair convention; caller quadrature=qf",
            },
        ),
    )
    dataset = from_interop(normalized)
    dataset.meta.stage = "acquisition"
    dataset.meta.experiment = params.get("experiment")
    dataset.filename = path
    dataset.history = (
        "Read uniform F1/F2 acquisition with caller-declared QF quadrature"
    )
    return dataset
