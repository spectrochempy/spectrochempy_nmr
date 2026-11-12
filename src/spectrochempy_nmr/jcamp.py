"""JCAMP-DX export of processed real 1D NMR spectra."""

from __future__ import annotations

import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from spectrochempy import Coord
from spectrochempy import NDDataset
from spectrochempy import ur


def _float64(values, field):
    values = np.asarray(values)
    if values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)):
        raise ValueError(f"{field} must contain finite real numbers")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        converted = values.astype(np.float64)
        if not np.array_equal(values, converted.astype(values.dtype)):
            raise ValueError(f"{field} must be representable in binary64")
    return converted


def _axis(dataset, size):
    """Validate the stored grid before Coord applies its display rounding."""
    coord = dataset.coord(dataset.dims[0])
    if not isinstance(coord, Coord) or coord.units not in (ur.Hz, ur.ppm):
        raise ValueError("JCAMP-DX requires a coordinate in Hz or ppm")
    values = _float64(coord._data, "coordinates")
    if values.shape != (size,):
        raise ValueError("the coordinate must contain one value per intensity")
    first, last = float(values[0]), float(values[-1])
    span = last - first
    step = span / (size - 1)
    if step == 0 or not np.isfinite(step):
        raise ValueError("the coordinate must have a finite nonzero step")
    monotone = values[1:] > values[:-1] if step > 0 else values[1:] < values[:-1]
    if not np.all(monotone):
        raise ValueError("the coordinate must be strictly monotone")
    expected = np.linspace(first, last, size)
    magnitude = max(abs(first), abs(last), abs(span))
    ulp = magnitude - np.nextafter(magnitude, 0.0)
    tolerance = np.minimum(4 * ulp, abs(step) * 1e-6)
    if not np.all(np.abs(values - expected) <= tolerance):
        raise ValueError("the coordinate must be regularly spaced")
    return coord, first, last, step


def _declared_nucleus(dataset):
    """Resolve a single observed isotope from source metadata."""
    meta = dataset.meta
    source = meta.get("source_metadata") or {}
    labels = source.get(".OBSERVENUCLEUS")
    if labels is None:
        procpar = meta.get("source_procpar")
        if procpar is None:
            procpar = source.get("procpar")
        if procpar is not None or "agilent" in (dataset.origin, meta.get("origin")):
            labels = (procpar or {}).get("tn", {}).get("values")
        else:
            labels = meta.get("nucleus")
            if labels is None:
                labels = meta.get("nuc1")
    if isinstance(labels, (list, tuple)) and len(labels) == 1:
        label = labels[0]
        if isinstance(label, str):
            return "^" + label.strip().removeprefix("^")
    return None


def _observation(dataset, coord, frequency, nucleus):
    if frequency is None:
        frequency = coord.meta.get("acquisition_frequency")
    if hasattr(frequency, "to"):
        frequency = frequency.to(ur.MHz).magnitude
    if (
        np.asarray(frequency).dtype.kind not in "iuf"
        or np.ndim(frequency) != 0
        or not np.isfinite(frequency := float(frequency))
        or frequency <= 0
    ):
        raise ValueError("a positive finite observation frequency in MHz is required")
    if nucleus is None:
        nucleus = _declared_nucleus(dataset)
    if not isinstance(nucleus, str) or not re.fullmatch(
        r"\^[1-9][0-9]*[A-Z][a-z]?", nucleus
    ):
        raise ValueError("an observation nucleus such as ^1H or ^13C is required")
    return frequency, nucleus


def _affn(value):
    """Use plain decimals when they fit the data line for TopSpin."""
    scientific = format(value, ".17g")
    if "e" in scientific:
        decimal = np.format_float_positional(value, unique=True, trim="-")
        if len(decimal) <= 26:
            return decimal
    return scientific


def write_jcamp(
    dataset: NDDataset,
    path: str | Path,
    *,
    observation_frequency=None,
    nucleus: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a real processed 1D NMR spectrum to a local JCAMP-DX file.

    The spectrum must have finite, unmasked intensities with ``dataset.units
    is None`` and a regular Hz or ppm coordinate. An explicit frequency is
    in MHz unless supplied as a quantity. Missing observation parameters are read from the
    dataset when available. Existing files require ``overwrite=True``.
    """
    if not isinstance(dataset, NDDataset):
        raise TypeError("dataset must be an NDDataset")
    if dataset.ndim != 1 or dataset.size < 2:
        raise ValueError("JCAMP-DX requires a 1D spectrum with at least two points")
    if dataset.meta.get("data_domain") == "time":
        raise ValueError("JCAMP-DX export requires a processed frequency spectrum")
    if dataset.units is not None or np.any(dataset.mask):
        raise ValueError("JCAMP-DX requires unmasked intensities without units")
    data = _float64(dataset.data, "intensities")
    coord, first, last, step = _axis(dataset, data.size)
    frequency, nucleus = _observation(dataset, coord, observation_frequency, nucleus)
    path = Path(path)
    if path.suffix.lower() not in {".dx", ".jdx"}:
        raise ValueError("JCAMP-DX output must have a .dx or .jdx extension")
    title = dataset.name.strip() or "NMR spectrum"
    if not title.isascii() or not title.isprintable() or "##" in title or "$$" in title:
        raise ValueError("JCAMP-DX title must be printable ASCII without ## or $$")
    if not overwrite and (path.exists() or path.is_symlink()):
        raise FileExistsError(path)
    xunits = "HZ" if coord.units == ur.Hz else "PPM"
    header = (
        f"##TITLE={title}\n"
        "##JCAMP-DX=5.01\n"
        "##DATA TYPE=NMR SPECTRUM\n"
        "##DATA CLASS=XYDATA\n"
        "##NUM DIM=1\n"
        f"##.OBSERVE FREQUENCY={frequency:.17g}\n"
        f"##.OBSERVE NUCLEUS={nucleus}\n"
        f"##XUNITS={xunits}\n"
        "##YUNITS=ARBITRARY UNITS\n"
        "##XFACTOR=1\n"
        "##YFACTOR=1\n"
        f"##FIRSTX={first:.17g}\n"
        f"##LASTX={last:.17g}\n"
        f"##NPOINTS={data.size}\n"
        f"##FIRSTY={data[0]:.17g}\n"
        "##XYDATA=(X++(Y..Y))\n"
    )
    with TemporaryDirectory(prefix=".jcamp-", dir=path.parent) as directory:
        staged = Path(directory) / path.name
        with staged.open("w", encoding="ascii", newline="\n") as stream:
            stream.write(header)
            for index in range(0, data.size, 2):
                intensities = " ".join(
                    _affn(value) for value in data[index : index + 2]
                )
                stream.write(f"{_affn(first + index * step)} {intensities}\n")
            stream.write("##END=\n")
        if overwrite:
            staged.replace(path)
        else:
            path.hardlink_to(staged)
    return path
