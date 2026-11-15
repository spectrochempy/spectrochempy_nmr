"""Exchange self-contained CSDF files for processed NMR spectra."""

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from spectrochempy import Coord
from spectrochempy import NDDataset
from spectrochempy import ur

_SI_PREFIX = r"(?:da|[qryzafpnumcdhkMGTPEZYRQ])?"
_SI_FREQUENCY = re.compile(_SI_PREFIX + "Hz")
_FREQUENCY_QUANTITY = re.compile(
    r"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
    r"\s*(\(?[A-Za-z]+\)?)"
)
_SI_FACTOR = re.compile(
    _SI_PREFIX
    + r"(?:m|s|g|A|K|mol|cd|rad|sr|Hz|N|Pa|J|W|C|V|F|Ohm|S|Wb|T|H|lm|lx|Bq|Gy|Sv|kat)"
    + r"(?:\^-?[0-9]+)?"
)
_DTYPES = {
    "float32": "<f4",
    "float64": "<f8",
    "complex64": "<c8",
    "complex128": "<c16",
}
_NUMERIC_TYPES = {np.dtype(dtype).name: name for name, dtype in _DTYPES.items()}


@dataclass(frozen=True)
class _Axis:
    values_hz: np.ndarray
    origin_offset_hz: float | None
    label: str | None


def read_csdf(path):
    """Read a processed 1D or 2D NMR spectrum from a self-contained CSDF file.

    Coordinates are expressed in Hz. CSDM origin offsets are stored in
    each coordinate's `meta.csdf_origin_offset_hz`.
    """
    values, axes, unit = _read_csdf(path)
    dataset = NDDataset(
        values.copy(),
        dtype=values.dtype,
        units=unit.replace("Ohm", "Ω") if unit else None,
        name=Path(path).stem,
        origin="CSDF",
    )
    coordinates = []
    for axis in axes:
        coord = Coord(
            axis.values_hz,
            units="Hz",
            title=axis.label,
            rounding=False,
            sigdigits=15,
        )
        if axis.origin_offset_hz is not None:
            coord.meta.csdf_origin_offset_hz = axis.origin_offset_hz
        coordinates.append(coord)
    dataset.set_coordset(*coordinates)
    for dim in dataset.dims:
        dataset.coord(dim)._rounding = False
    dataset.meta.data_domain = "frequency"
    dataset.meta.isfreq = [True] * dataset.ndim
    if dataset.ndim == 1:
        dataset.meta.iscomplex = [np.iscomplexobj(values)]
    return dataset


def write_csdf(
    dataset: NDDataset, path: str | Path, *, overwrite: bool = False
) -> Path:
    """Write an unmasked processed 1D or 2D spectrum with linear frequency axes.

    Coordinate labels, signal units and CSDM origin offsets are included
    when available. Other dataset metadata are omitted.
    """
    if not isinstance(dataset, NDDataset):
        raise TypeError("dataset must be an NDDataset")
    path = Path(path)
    if path.suffix.lower() != ".csdf":
        raise ValueError("Expected a self-contained .csdf file")
    values = dataset.data
    if dataset.ndim not in (1, 2) or not dataset.size:
        raise ValueError("CSDF export requires a nonempty 1D or 2D spectrum")
    numeric_type = _NUMERIC_TYPES.get(values.dtype.name)
    if numeric_type is None or np.any(dataset.mask):
        raise ValueError("CSDF export requires unmasked float or complex data")
    dimensions = [
        _encode_dimension(dataset.coord(dim), count)
        for dim, count in zip(dataset.dims, values.shape, strict=True)
    ]
    signal = {
        "type": "internal",
        "quantity_type": "scalar",
        "numeric_type": numeric_type,
        "encoding": "base64",
        "components": [
            base64.b64encode(
                values.astype(_DTYPES[numeric_type], copy=False).tobytes()
            ).decode("ascii")
        ],
    }
    if dataset.units:
        unit = format(dataset.units, "~")
        if unit:
            signal["unit"] = _encode_signal_unit(unit)
    model = {
        "version": "1.0",
        "dimensions": dimensions[::-1],
        "dependent_variables": [signal],
    }
    with TemporaryDirectory(prefix=".csdf-", dir=path.parent) as directory:
        staged = Path(directory) / path.name
        with staged.open("w", encoding="utf-8") as stream:
            json.dump({"csdm": model}, stream, allow_nan=False)
            stream.write("\n")
        if overwrite:
            staged.replace(path)
        else:
            path.hardlink_to(staged)
    return path


def _encode_signal_unit(unit):
    """Serialize an SI unit with explicit multiplication between factors."""
    factors = unit.replace("μ", "u").replace("µ", "u").replace("Ω", "Ohm").split()
    if not all(_SI_FACTOR.fullmatch(factor) for factor in factors):
        raise ValueError(f"CSDF export requires portable SI signal units, got {unit!r}")
    encoded = "*".join(factors)
    return f"({encoded})" if encoded.startswith("E") else encoded


def _ascii_frequency_unit(unit):
    return unit.replace("μ", "u").replace("µ", "u")


def _encode_dimension(coord, count):
    if (
        not isinstance(coord, Coord)
        or coord.data is None
        or coord.units is None
        or not _SI_FREQUENCY.fullmatch(_ascii_frequency_unit(format(coord.units, "~")))
        or np.any(coord.mask)
    ):
        raise ValueError(
            "CSDF export requires an unmasked frequency coordinate per axis"
        )
    values = (
        np.asarray(coord._data, dtype=np.float64) * (1 * coord.units).to("Hz").magnitude
    )
    if values.shape != (count,) or not np.all(np.isfinite(values)):
        raise ValueError(
            "CSDF export requires finite coordinates matching the data shape"
        )
    increment = (values[-1] - values[0]) / (count - 1) if count > 1 else 1.0
    expected = values[0] + np.arange(count) * increment
    magnitude = np.max(np.abs(values))
    tolerance = 8 * (magnitude - np.nextafter(magnitude, 0.0))
    if (
        not np.isfinite(increment)
        or increment == 0
        or not np.all(np.diff(values) * np.sign(increment) > 0)
        or not np.all(np.diff(expected) * np.sign(increment) > 0)
        or not np.allclose(values, expected, rtol=0, atol=tolerance)
    ):
        raise ValueError("CSDF export requires a linear frequency coordinate")
    dimension = {
        "type": "linear",
        "count": count,
        "increment": f"{increment:.17g} Hz",
        "coordinates_offset": f"{values[0]:.17g} Hz",
    }
    if coord.title and coord.title != "<untitled>":
        dimension["label"] = coord.title
    origin = coord.meta.get("csdf_origin_offset_hz")
    if origin is not None:
        if not np.isfinite(origin):
            raise ValueError("CSDF origin offsets must be finite")
        dimension["origin_offset"] = f"{origin:.17g} Hz"
    return dimension


def _frequency(value):
    if not isinstance(value, str):
        raise ValueError("Expected a CSDF frequency quantity")
    match = _FREQUENCY_QUANTITY.fullmatch(_ascii_frequency_unit(value.strip()))
    if match is None:
        raise ValueError("Expected a CSDF quantity with frequency units")
    magnitude, unit = match.groups()
    if unit.startswith("(") and unit.endswith(")"):
        unit = unit[1:-1]
    if not _SI_FREQUENCY.fullmatch(unit):
        raise ValueError("Expected Hz or an SI-prefixed Hz unit")
    frequency = (float(magnitude) * ur.Unit(unit)).to("Hz").magnitude
    if not np.isfinite(frequency):
        raise ValueError("CSDF frequencies must be finite")
    return frequency


def _read_csdf(path):
    """Return a signal, axes in NumPy order and the declared signal unit."""
    path = Path(path)
    if path.suffix.lower() != ".csdf":
        raise ValueError("Expected a self-contained .csdf file")
    model = json.loads(path.read_text(encoding="utf-8"))["csdm"]
    dimensions = model["dimensions"]
    variables = model["dependent_variables"]
    if model.get("version") != "1.0" or len(dimensions) not in (1, 2):
        raise ValueError("Expected CSDF 1.0 with one or two frequency dimensions")
    if len(variables) != 1:
        raise ValueError("Expected exactly one CSDF dependent variable")

    signal = variables[0]
    dtype = _DTYPES.get(signal.get("numeric_type"))
    components = signal.get("components", [])
    if (
        signal.get("type") != "internal"
        or signal.get("quantity_type") != "scalar"
        or signal.get("encoding") != "base64"
        or dtype is None
        or not isinstance(components, list)
        or len(components) != 1
        or signal.get("sparse_sampling") is not None
    ):
        raise ValueError("Expected one internal, dense, base64 scalar signal")
    payload = base64.b64decode(components[0], validate=True)
    values = np.frombuffer(payload, dtype=dtype)

    shape = []
    grids = []
    for dimension in reversed(dimensions):
        count = dimension.get("count")
        if (
            dimension.get("type") != "linear"
            or not isinstance(count, int)
            or isinstance(count, bool)
            or not 0 < count <= values.size
            or not isinstance(dimension.get("complex_fft", False), bool)
        ):
            raise ValueError("Expected a linear frequency axis with a valid count")
        shape.append(count)
        increment = _frequency(dimension["increment"])
        offset = _frequency(dimension.get("coordinates_offset", "0 Hz"))
        if increment == 0:
            raise ValueError("CSDF frequency increment must be nonzero")
        origin = dimension.get("origin_offset")
        zero_index = count // 2 if dimension.get("complex_fft", False) else 0
        grids.append(
            (count, increment, offset, zero_index, origin, dimension.get("label"))
        )

    if values.size != int(np.prod(shape, dtype=object)):
        raise ValueError("CSDF payload size does not match the declared shape")

    axes = []
    for count, increment, offset, zero_index, origin, label in grids:
        coordinates = offset + (np.arange(count) - zero_index) * increment
        steps = np.diff(coordinates)
        if not np.all(np.isfinite(coordinates)) or not np.all(
            steps > 0 if increment > 0 else steps < 0
        ):
            raise ValueError("CSDF frequency coordinates must be finite and monotonic")
        axes.append(
            _Axis(coordinates, None if origin is None else _frequency(origin), label)
        )
    return values.reshape(shape), tuple(axes), signal.get("unit")
