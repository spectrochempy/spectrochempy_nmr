# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
JEOL JDF file importers.

This module provides functionality to read JEOL JDF (JEOL Data Format)
NMR data files.

Functions
---------
- read_jeol : Main entry point for reading JEOL JDF files

Notes
-----
Supports 1D and 2D data only.  3D and higher-dimensional datasets
raise ``NotImplementedError``.

"""

__all__ = ["read_jeol"]

import contextlib
import re
from datetime import datetime

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.core.units import ur
from spectrochempy.utils._logging import warning_
from spectrochempy.utils.meta import Meta
from spectrochempy_nmr.extern.nmrglue._jeol import read_jeol as _read_jeol_raw
from spectrochempy_nmr.readers._metadata import apply_reader_metadata_options

# ======================================================================================
# Public entry point
# ======================================================================================


def read_jeol(*paths, **kwargs):
    r"""
    Open JEOL JDF NMR spectra.

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` objects or valid urls, optional
        The data source(s).  Can be:

        - a path to a ``.jdf`` file
        - a list of paths (datasets are merged if ``merge=True``)

    **kwargs : keyword parameters, optional
        See Other Parameters.

    Returns
    -------
    object : `NDDataset` or `ScpObjectList` of `NDDataset`
        The returned dataset(s).

    Other Parameters
    ----------------
    content : `bytes`, optional
        Raw bytes content instead of a filename.
    description : `str`, optional
        A custom description.
    directory : `~pathlib.Path`, optional
        Base directory for resolving relative paths.
    merge : `bool`, optional, default: ``False``
        If ``True`` and several filenames are provided, merge into a
        single dataset.
    origin : str, optional
        Override the origin label (default ``'jeol'``).

    See Also
    --------
    read : Generic reader inferring protocol from filename extension.

    Examples
    --------
    Reading a single JEOL JDF file:

    >>> scp.nmr.read('path/to/experiment.jdf')  # doctest: +SKIP

    """
    kwargs["filetypes"] = ["JEOL JDF files (*.jdf)"]
    kwargs["protocol"] = ["jeol"]
    importer = _JeolImporter()
    return importer(*paths, **kwargs)


# ======================================================================================
# Private reader implementation
# ======================================================================================


class _JeolImporter(Importer):
    """Keep potentially stacked reads on the historical dataset contract."""

    def _switch_protocol(self, key, files, **kwargs):
        kwargs["_jeol_single_acquisition"] = (
            not kwargs.get("merge", True)
            or sum(len(group) for group in self.files.values()) == 1
        )
        return super()._switch_protocol(key, files, **kwargs)


def _jeol_axis_type_to_encoding(axis_type_str, *, axis: int, ndim: int):
    """
    Map a JEOL ``data_axis_type`` string to a canonical encoding.

    JEOL uses axis-type labels such as ``complex`` and ``real_complex`` for
    both 1D and 2D data.  In 1D those values describe direct-dimension complex
    quadrature and should therefore be normalized to a vendor-independent
    direct encoding (``QSIM``), not to 2D indirect encodings such as
    ``STATES``.
    """
    axis_type = str(axis_type_str).lower()

    if axis_type == "real":
        return "QF", False
    if axis_type == "tppi":
        return "TPPI", True
    if axis_type == "complex":
        if ndim == 1 or axis == ndim - 1:
            return "QSIM", True
        return "STATES", True
    if axis_type == "real_complex":
        if ndim == 1 or axis == ndim - 1:
            return "QSIM", True
        return "STATES-TPPI", True
    return "unknown", False


def _read_normalized_jeol_fid(path):
    from inspect import signature  # noqa: PLC0415

    from nmrglue import interop  # noqa: PLC0415

    from spectrochempy_nmr.interop import from_interop  # noqa: PLC0415

    normalize = interop.normalize_jeol_fid
    if "legacy_presentation" not in signature(normalize).parameters:
        return None
    try:
        stored = interop.read_jeol_fid(path)
        result = normalize(stored, conjugation="conjugate", legacy_presentation=True)
    except (ValueError, OSError):
        return None

    dataset = from_interop(result)
    acquisition = dataset.meta.acquisition
    display = acquisition["legacy_presentation"]
    width = acquisition["spectral_width_hz"]
    observation = acquisition["observation_frequency_mhz"]
    dataset.x.ito("us")
    dataset.x.title = "F1 acquisition time"
    dataset.x.meta["acquisition_frequency"] = observation * ur.MHz
    dataset.meta.update(
        Meta(
            datatype="FID",
            pathname=str(path),
            filename=path.name,
            origin="jeol",
            endian=display["data_endianness"],
            data_format=display["storage_format"],
            data_type=display["storage_dtype"],
            sw=[width],
            sw_h=[width * ur.Hz],
            sfrq=[observation],
            nucleus=[acquisition.get("nucleus")],
            iscomplex=[True],
            offset=[acquisition.get("carrier_offset_ppm")],
            ns=display["scan_count"],
            solvent=display["solvent"],
            sample=display["sample"],
            experiment=display["pulse_program"],
            at=display["acquisition_time"],
            d1=display["relaxation_delay"],
            pw=display["pulse_width"],
        )
    )
    return _finish_jeol_dataset(dataset, path, display["acquisition_date"])


def _finish_jeol_dataset(dataset, path, creation):
    dataset.meta.readonly = True
    dataset.units = "count"
    dataset.title = "intensity"
    dataset.name = f"{path.stem} ({dataset.meta.datatype})"
    dataset.filename = path.parent
    if creation:
        with contextlib.suppress(ValueError, TypeError):
            dataset.acquisition_date = datetime(*creation)
    dataset.history = f"Imported JEOL JDF file {path.name}"
    return dataset


@_importer_method
def _read_jdf(*args, **kwargs):
    dataset, path = args

    if not path.exists():
        warning_(f"File not found: {path}")
        return None

    normalized = (
        _read_normalized_jeol_fid(path)
        if kwargs.get("_jeol_single_acquisition") and type(dataset) is NDDataset
        else None
    )
    if normalized is not None:
        apply_reader_metadata_options(normalized, kwargs, default_origin="jeol")
        return normalized

    try:
        dic, data = _read_jeol_raw(str(path))
    except Exception as exc:
        warning_(f"Failed to read JEOL file {path}: {exc}")
        return None

    header = dic["header"]
    params = dic["parameters"]

    # Determine ndim from data shape
    ndim = data.ndim
    if ndim > 2:
        msg = f"JEOL datasets with {ndim} dimensions are not supported (max 2D)."
        raise NotImplementedError(msg)

    # Axis prefixes: x, y, z, ...
    axis_prefixes = ["x", "y", "z", "a", "b", "c", "d", "e"]

    # Build metadata
    meta = Meta()

    # Basic info
    meta.datatype = f"{ndim}D" if ndim >= 2 else "FID"
    meta.pathname = str(path)
    meta.filename = path.name
    meta.origin = "jeol"

    # Header parameters
    meta.endian = header.get("endian")
    meta.data_format = header.get("data_format")
    meta.data_type = header.get("data_type")

    # Per-dimension lists
    meta.sw = [None] * ndim
    meta.sfrq = [None] * ndim
    meta.td = list(data.shape)
    meta.nucleus = [None] * ndim
    meta.encoding = [None] * ndim
    meta.iscomplex = [False] * ndim
    meta.isfreq = [False] * ndim
    meta.offset = [None] * ndim

    for i in range(ndim):
        prefix = axis_prefixes[i]

        # Spectral width in Hz
        sw = params.get(f"{prefix}_sweep")
        meta.sw[i] = sw

        # Spectrometer frequency in MHz (JEOL stores in Hz)
        freq_hz = params.get(f"{prefix}_freq")
        if freq_hz is not None:
            meta.sfrq[i] = freq_hz / 1e6

        # Nucleus
        meta.nucleus[i] = params.get(f"{prefix}_domain")

        # Offset in ppm
        meta.offset[i] = params.get(f"{prefix}_offset")

        # Encoding and iscomplex from axis_type
        axis_type = header.get("data_axis_type", [None] * 8)[i]
        if axis_type is not None:
            enc, ic = _jeol_axis_type_to_encoding(axis_type, axis=i, ndim=ndim)
            meta.encoding[i] = enc
            meta.iscomplex[i] = ic

    # Spectral width in Hz with units
    meta.sw_h = [None] * ndim
    for i in range(ndim):
        if meta.sw[i] is not None:
            meta.sw_h[i] = meta.sw[i] * ur.Hz

    # Number of scans
    meta.ns = params.get("total_scans", params.get("scans", 1))

    # Solvent
    meta.solvent = params.get("solvent")

    # Sample
    meta.sample = params.get("sample")

    # Experiment / pulse program
    meta.experiment = params.get("experiment")

    # Acquisition time
    meta.at = params.get("x_acq_time")

    # Relaxation delay
    meta.d1 = params.get("relaxation_delay")

    # 90-degree pulse width
    meta.pw = params.get("x90")

    # Build coordinates
    coords = []

    def _make_coord(axis, size):
        """Build a single coordinate for the given axis."""
        sw_hz = meta.sw_h[axis]
        freq_mhz = meta.sfrq[axis]
        nuc = meta.nucleus[axis]
        offset_ppm = meta.offset[axis]

        if not meta.isfreq[axis]:
            if sw_hz is not None and float(sw_hz.magnitude) > 0:
                dw = (1.0 / sw_hz).to("us")
                coordpoints = np.arange(size)
                coord = Coord(coordpoints * dw, title=f"F{axis + 1} acquisition time")
            else:
                coord = Coord(np.arange(size), title=f"F{axis + 1} acquisition time")

            if freq_mhz is not None:
                coord.meta["acquisition_frequency"] = freq_mhz * ur.MHz
            return coord

        if sw_hz is not None and freq_mhz is not None and size > 1:
            sizem = max(size - 1, 1)
            deltaf = -float(sw_hz.magnitude) / sizem

            # Use offset to set the carrier position
            if offset_ppm is not None:
                first = (
                    float(freq_mhz) * 1e6
                    + float(offset_ppm) * float(freq_mhz) * 1e3
                    - deltaf * sizem / 2.0
                )
            else:
                first = float(freq_mhz) * 1e6 - deltaf * sizem / 2.0

            coordpoints = np.arange(size) * deltaf + first
            coord = Coord(coordpoints)
            coord.meta["acquisition_frequency"] = freq_mhz * ur.MHz
            coord.ito("ppm")

            if nuc:
                regex = r"([^a-zA-Z]+)([a-zA-Z]+)"
                m = re.match(regex, str(nuc))
                nucleus = rf"$^{{{m[1]}}}{m[2]}$" if m else str(nuc)
            else:
                nucleus = ""
            coord.title = rf"$\delta\ {nucleus}$"
        else:
            coord = Coord(np.arange(size), title=f"F{axis + 1}")

        return coord

    for axis in range(ndim):
        coords.append(_make_coord(axis, meta.td[axis]))

    # Set data
    if ndim >= 2 and meta.iscomplex[-2]:
        from spectrochempy_nmr.processing.fft_validation import (
            _require_fft_prevalidation,
        )

        _require_fft_prevalidation()
        meta.encoding[-2] = "UNVALIDATED:JEOL-" + str(meta.encoding[-2])
        meta.normalization_state = "storage-decoded"

    dataset.data = data

    # Apply metadata
    dataset.meta.update(meta)
    dataset.set_coordset(*tuple(coords))
    apply_reader_metadata_options(dataset, kwargs, default_origin="jeol")
    creation = header.get("creation_time")
    if creation:
        creation = (
            creation.get("year", 1970),
            creation.get("month", 1),
            creation.get("day", 1),
        )
    return _finish_jeol_dataset(dataset, path, creation)
