# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Agilent/Varian NMR file importers.

This module provides functionality to read Agilent/Varian NMR data files
(FID binary + procpar parameter files).

Functions
---------
- read_agilent : Main entry point for reading Agilent/Varian files

Notes
-----
Supports 1D FID and multidimensional data.  Only reading is implemented;
no write or low-memory variants are provided.

"""

__all__ = ["read_agilent"]

from datetime import datetime
from inspect import signature

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.core.units import ur
from spectrochempy.utils._logging import info_
from spectrochempy.utils._logging import warning_
from spectrochempy.utils.exceptions import SpectroChemPyError
from spectrochempy.utils.meta import Meta
from nmrglue.fileio.varian import read as read_varian
from spectrochempy_nmr.processing.fft_validation import _require_fft_prevalidation
from spectrochempy_nmr.readers._channels import pair_traces
from spectrochempy_nmr.readers._channels import set_channel_data
from spectrochempy_nmr.readers._metadata import apply_reader_metadata_options

# ======================================================================================
# Valid Agilent parameter names for metadata extraction
# ======================================================================================

# procpar parameter names → (unit_string, is_direct)
# Direct dimension parameters: sw, sfrq/dfrq, np, at, dn/tn
# Indirect dimension parameters: ni, ni2, ni3, sw2/sfrq2, etc.
agilent_valid_meta = [
    ("sw", "Hz"),
    ("sfrq", "MHz"),
    ("dfrq", "MHz"),
    ("dn", ""),
    ("tn", ""),
    ("np", ""),
    ("ni", ""),
    ("ni2", ""),
    ("ni3", ""),
    ("nt", ""),
    ("ne", ""),
    ("ns", ""),
    ("sw2", "Hz"),
    ("sfrq2", "MHz"),
    ("dfrq2", "MHz"),
    ("dn2", ""),
    ("tn2", ""),
    ("sw3", "Hz"),
    ("sfrq3", "MHz"),
    ("dfrq3", "MHz"),
    ("dn3", ""),
    ("tn3", ""),
    ("sw4", "Hz"),
    ("sfrq4", "MHz"),
    ("dfrq4", "MHz"),
    ("dn4", ""),
    ("tn4", ""),
    ("at", "s"),
    ("rt", "s"),
    ("t1", "s"),
    ("d1", "s"),
    ("pw", "us"),
    ("p1", "us"),
    ("ro", ""),
    ("tp", ""),
    ("phase", ""),
    ("phase2", ""),
    ("phase3", ""),
    ("array", ""),
    ("pulprog", ""),
    ("seqcon", ""),
    ("filnam", ""),
    ("date", ""),
]


def _extract_procpar_value(pdic, key, default=None):
    """Extract the first value from a procpar parameter, converting to float if possible."""
    if key not in pdic:
        return default
    values = pdic[key].get("values", [])
    if not values:
        return default
    val = values[0]
    try:
        return float(val)
    except (ValueError, TypeError):
        return val


def _extract_procpar_string(pdic, key, default=None):
    """Extract the first string value from a procpar parameter."""
    if key not in pdic:
        return default
    values = pdic[key].get("values", [])
    return values[0] if values else default


def _extract_procpar_values(pdic, key):
    """Extract a procpar value list, treating malformed values as empty."""
    try:
        values = pdic.get(key, {}).get("values", [])
    except (AttributeError, TypeError):
        return []
    return values if isinstance(values, (list, tuple)) else []


def _extract_procpar_integer(pdic, key, default=None):
    """Extract a finite integer procpar value without truncation."""
    try:
        values = pdic.get(key, {}).get("values", [])
        if not isinstance(values, (list, tuple)) or len(values) != 1:
            return default
        value = values[0]
        if isinstance(value, (list, tuple, np.ndarray)):
            return default
    except (AttributeError, TypeError, ValueError, OverflowError):
        return default
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not np.isfinite(value) or not value.is_integer():
        return default
    return int(value)


def _detect_agilent_encoding(pdic, ndim):
    """
    Detect the quadrature encoding for each dimension from procpar.

    Returns
    -------
    tuple of str
        Encoding per dimension.  One of 'QF', 'TPPI', 'STATES', 'STATES-TPPI'.
    """
    encodings = []

    # Direct dimension: check if data is complex
    np_val = _extract_procpar_integer(pdic, "np")
    is_direct_complex = np_val is not None and np_val % 2 == 0 and np_val > 0
    encodings.append("QF" if not is_direct_complex else "QSIM")

    # Indirect dimensions
    phase_keys = ["phase", "phase2", "phase3"]
    for i in range(ndim - 1):
        if i < len(phase_keys):
            phase_vals = _extract_procpar_values(pdic, phase_keys[i])
            if len(phase_vals) >= 2:
                encodings.append("STATES")
            elif len(phase_vals) == 1:
                encodings.append("TPPI")
            else:
                encodings.append("QF")
        else:
            encodings.append("QF")

    return tuple(encodings)


def _has_unvalidated_states_phase(pdic):
    """Return whether a 2D phase field needs an explicit States diagnostic."""
    if "phase" not in pdic:
        ni = _extract_procpar_integer(pdic, "ni")
        arraydim = _extract_procpar_integer(pdic, "arraydim")
        return ni is not None and arraydim == 2 * ni
    phase_entry = pdic["phase"]
    if not isinstance(phase_entry, dict) or "values" not in phase_entry:
        return True
    phase_values = phase_entry["values"]
    if not isinstance(phase_values, (list, tuple)):
        return True
    if len(phase_values) == 0 or len(phase_values) >= 2:
        return True
    if isinstance(phase_values[0], (list, tuple, np.ndarray)):
        return True
    try:
        value = float(phase_values[0])
    except (TypeError, ValueError):
        return True
    return not np.isfinite(value) or not value.is_integer()


def _reconstruct_agilent_states(data):
    """Separate successive Agilent States traces into two complex channels."""
    return pair_traces(data)


def _validate_agilent_states_profile(pdic, dic, data):
    """Validate the Agilent 2D States profile before row reconstruction.

    The row-pairing convention is established only for the standard
    ``phase=1,2`` acquisition with ``arraydim=2*ni`` and one trace per block.
    Return a reason instead of guessing when the procpar or binary dimensions
    describe another acquisition.
    """
    if data.ndim != 2:
        return False, f"expected 2D data, received {data.ndim}D"

    phase_values = _extract_procpar_values(pdic, "phase")

    def _finite_integer_value(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            return None
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(value) or not value.is_integer():
            return None
        return int(value)

    phase = [_finite_integer_value(value) for value in phase_values]
    if any(value is None for value in phase):
        return False, f"unsupported phase values {phase_values!r}"
    if phase != [1, 2]:
        return False, f"expected phase=[1, 2], received {phase_values!r}"

    ni = _extract_procpar_integer(pdic, "ni")
    arraydim = _extract_procpar_integer(pdic, "arraydim")
    np_raw = _extract_procpar_integer(pdic, "np")
    if ni is None or arraydim is None or np_raw is None:
        return False, "ni, arraydim and np must be finite integers"

    expected_traces = 2 * ni
    if ni <= 0 or arraydim != expected_traces:
        return False, f"expected arraydim=2*ni, received arraydim={arraydim}, ni={ni}"
    if data.shape[0] != expected_traces:
        return False, (
            f"expected {expected_traces} traces from 2*ni, received {data.shape[0]}"
        )
    if np_raw <= 0 or np_raw % 2 or data.shape[1] != np_raw // 2:
        return False, f"np={np_raw} is inconsistent with data shape {data.shape}"
    if "ntraces" not in dic:
        return False, "missing ntraces metadata"
    if isinstance(dic["ntraces"], (list, tuple, np.ndarray)):
        return False, "ntraces must be a scalar integer"
    if not isinstance(dic["ntraces"], (int, np.integer)) or dic["ntraces"] != 1:
        return False, f"expected ntraces=1, received {dic['ntraces']}"

    return True, None


# ======================================================================================
# Public entry point
# ======================================================================================


def read_agilent(*paths, **kwargs):
    r"""
    Open Agilent/Varian NMR spectra.

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` objects or valid urls, optional
        The data source(s).  Can be:

        - a path to a directory containing ``fid`` and ``procpar`` files
        - a path to the ``fid`` file itself
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
        Override the origin label (default ``'agilent'``).
    quadrature : {"auto", "states", "states-tppi"}, optional
        Choose the F1 convention for qualified paired 2D FIDs. Automatic
        selection requires a recognized pulse program source.
    pulse_program_source : path-like, optional
        Source of the pulse program used for this acquisition, for automatic
        selection. By default the provider checks for a known adjacent source.

    See Also
    --------
    read : Generic reader inferring protocol from filename extension.

    Examples
    --------
    Reading a single Agilent FID directory:

    >>> scp.nmr.read('path/to/agilent_1d/')  # doctest: +SKIP

    """
    if kwargs.get("quadrature", "auto") not in ("auto", "states", "states-tppi"):
        raise ValueError("quadrature must be auto, states or states-tppi")
    if (
        kwargs.get("pulse_program_source") is not None
        and kwargs.get("quadrature", "auto") != "auto"
    ):
        raise ValueError("pulse_program_source requires automatic quadrature")
    kwargs["filetypes"] = ["Agilent/Varian files (fid, procpar)"]
    kwargs["protocol"] = ["agilent"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ======================================================================================
# Private reader implementation
# ======================================================================================


def _resolve_agilent_directory_target(filename, **kwargs):
    """Resolve an Agilent directory to the fid file within it."""
    protocol = kwargs.get("protocol")
    if protocol is None:
        return None
    if isinstance(protocol, str):
        protocol = [protocol]
    if "agilent" not in protocol:
        return None

    # If it's already a fid file, no resolution needed
    if filename.name == "fid":
        return None

    # Check if this directory contains a fid file
    fid_path = filename / "fid"
    if fid_path.exists():
        return [fid_path]

    return None


def _infer_agilent_filetype_key(filename, **kwargs):
    """Return the Agilent filetype key for fid files."""
    if filename.name == "fid" and filename.parent.exists():
        procpar = filename.parent / "procpar"
        if procpar.exists():
            return ".agilent"
    return None


def _ensure_agilent_filetype_registered() -> None:
    """Register the plugin-owned Agilent key in the legacy importer registry."""
    from spectrochempy.core.readers.filetypes import registry  # noqa: PLC0415

    known = {name for name, _description in registry.filetypes}
    if "agilent" not in known:
        registry.register_filetype(
            "agilent",
            "Agilent/Varian NMR fid and procpar files",
        )


def _read_normalized_agilent_fid(directory, path):
    from inspect import signature  # noqa: PLC0415

    from nmrglue import interop  # noqa: PLC0415

    from ..interop import from_interop  # noqa: PLC0415

    normalize = interop.normalize_agilent_fid
    if "legacy_presentation" not in signature(normalize).parameters:
        return None
    try:
        stored = interop.read_agilent_fid(directory, single_trace_only=True)
        result = normalize(stored, legacy_presentation=True)
        dataset = from_interop(result)
    except ValueError:
        return None
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
            filename=directory.parent,
            origin="agilent",
            nblocks=display["nblocks"],
            ntraces=display["ntraces"],
            np_raw=int(acquisition["np_raw"]),
            source_procpar=dataset.meta.source_metadata["procpar"],
            sw=[width],
            sw_h=[width * ur.Hz],
            sfrq=[observation],
            dfrq=[display["decoupler_frequency_mhz"]],
            nucleus=[display["reported_nucleus"]],
            iscomplex=[True],
            ns=display["ns"],
            ne=display["ne"],
            ns_norm=1.0,
            pulprog=display["pulse_program"],
            date_raw=display["date_raw"],
            at=display["acquisition_time_s"],
        )
    )
    return dataset


def _read_normalized_agilent_2d(
    directory, path, profile, *, quadrature="auto", pulse_program_source=None
):
    from nmrglue import interop  # noqa: PLC0415

    from ..interop import from_interop  # noqa: PLC0415

    name = "fid_series" if profile == "series" else f"{profile}_2d"
    normalize = getattr(interop, f"normalize_agilent_{name}", None)
    if normalize is None:
        return None
    requested = quadrature != "auto" or pulse_program_source is not None
    options = {}
    if profile == "states":
        if "quadrature" in signature(normalize).parameters:
            options = {
                "quadrature": quadrature,
                "pulse_program_source": pulse_program_source,
            }
        elif requested:
            raise SpectroChemPyError(
                "Agilent quadrature selection requires a newer nmrglue-ng provider"
            )
    try:
        stored = interop.read_agilent_fid_storage(
            directory, **{f"{profile}_only": True}
        )
        result = normalize(stored, legacy_presentation=True, **options)
        dataset = from_interop(result)
    except (OSError, NotImplementedError) as exc:
        if requested:
            raise SpectroChemPyError(str(exc)) from exc
        raise
    except ValueError as exc:
        if requested:
            raise SpectroChemPyError(str(exc)) from exc
        return None

    acquisition = dataset.meta.acquisition
    selection = acquisition.get("quadrature_selection", {})
    mode = selection.get("mode")
    display = acquisition["legacy_presentation"]
    widths = display["spectral_widths_hz"]
    observation = display["observation_frequencies_mhz"]
    decoupler = display["decoupler_frequencies_mhz"]
    if mode is not None:
        widths = acquisition["spectral_width_hz"]
        observation = acquisition["observation_frequencies_mhz"]
    for index, (dim, width, frequency, fallback) in enumerate(
        zip(("y", "x"), widths, observation, decoupler, strict=True), 1
    ):
        coord = dataset.coordset[dim]
        if width is not None and mode is None:
            coord = Coord(coord.to("us").data, units="us")
        coord.title = f"F{index} acquisition time"
        display_frequency = frequency if mode is not None else frequency or fallback
        if display_frequency is not None:
            coord.meta["acquisition_frequency"] = display_frequency * ur.MHz
        dataset.coordset[dim] = coord

    states = profile == "states"
    prefix = [None] if states else []
    dataset.meta.update(
        Meta(
            datatype="2D",
            pathname=str(path),
            filename=directory.parent,
            origin="agilent",
            nblocks=acquisition.get("blocks", acquisition["storage_shape"][0]),
            ntraces=acquisition.get("traces_per_block", 1),
            np_raw=acquisition["storage_shape"][1],
            source_procpar=dataset.meta.source_metadata["procpar"],
            sw=[*prefix, *widths],
            sw_h=[
                *prefix,
                *(None if width is None else width * ur.Hz for width in widths),
            ],
            sfrq=[*prefix, *observation],
            dfrq=[*prefix, *decoupler],
            nucleus=[*prefix, *display["reported_nuclei"]],
            ns=display["ns"],
            ne=display["ne"],
            ns_norm=1.0,
            pulprog=display["pulse_program"],
            date_raw=display["date_raw"],
            at=display["acquisition_time_s"],
        )
    )
    if states:
        dataset.meta.update(
            Meta(
                encoding=(
                    dataset.meta.encoding
                    if mode is not None
                    else ["CHANNELS", "UNVALIDATED:AGILENT-STATES", "DIRECT:QSIM"]
                ),
                acquisition_encoding=((mode or "states").upper(), "QSIM"),
                storage_profile="agilent-phase-1-2-trace-pairs",
                seqfil=display["sequence"],
                phase=acquisition["phase"],
                ni=acquisition["physical_indirect_increments"],
                arraydim=acquisition["storage_shape"][0],
            )
        )
        if mode is not None:
            info_(f"Agilent F1 quadrature: {mode} ({selection['source']})")
        elif selection:
            info_(
                "Agilent F1 quadrature undetermined; specify quadrature to process F1"
            )
    elif profile == "tppi":
        dataset.meta.encoding = ["UNVALIDATED:AGILENT-TPPI", "QSIM"]
    return dataset


def _finalize_agilent_dataset(dataset, directory, options):
    dataset.meta.readonly = True
    dataset.units = "count"
    dataset.title = "intensity"
    apply_reader_metadata_options(dataset, options, default_origin="agilent")
    datatype = dataset.meta.datatype
    dataset.name = f"{directory.name} ({datatype})"
    dataset.filename = directory

    if dataset.meta.date_raw:
        import contextlib

        with contextlib.suppress(ValueError, TypeError):
            dataset.acquisition_date = datetime.fromisoformat(
                str(dataset.meta.date_raw)
            )

    dataset.history = f"Imported Agilent/Varian dataset {directory.name} ({datatype})"
    return dataset


@_importer_method
def _read_agilent(*args, **kwargs):
    dataset, path = args

    # Determine the directory containing fid and procpar
    data_dir = path.parent if path.name == "fid" else path

    fid_path = data_dir / "fid"
    procpar_path = data_dir / "procpar"

    if not fid_path.exists():
        warning_(f"No fid file found in {data_dir}")
        return None
    if not procpar_path.exists():
        warning_(f"No procpar file found in {data_dir}")
        return None

    quadrature = kwargs.get("quadrature", "auto")
    pulse_program_source = kwargs.get("pulse_program_source")
    if quadrature != "auto" or pulse_program_source is not None:
        candidate = _read_normalized_agilent_2d(
            data_dir,
            path,
            "states",
            quadrature=quadrature,
            pulse_program_source=pulse_program_source,
        )
        if candidate is None:
            raise SpectroChemPyError(
                "Quadrature selection requires a paired Agilent 2D FID"
            )
        return _finalize_agilent_dataset(candidate, data_dir, kwargs)

    candidate = _read_normalized_agilent_fid(data_dir, path)
    if candidate is not None:
        return _finalize_agilent_dataset(candidate, data_dir, kwargs)

    for profile in ("states", "tppi", "series"):
        candidate = _read_normalized_agilent_2d(data_dir, path, profile)
        if candidate is not None:
            return _finalize_agilent_dataset(candidate, data_dir, kwargs)

    # Compatibility presentation for profiles outside the normalized contracts.
    dic, data = read_varian(data_dir, legacy_layout=True)
    pdic = dic.get("procpar", {})

    # Determine ndim from data shape
    ndim = data.ndim
    if ndim > 2:
        raise NotImplementedError(
            "Agilent reader supports validated 1D and 2D storage only"
        )

    # Build metadata
    meta = Meta()

    # Basic info
    meta.datatype = f"{ndim}D" if ndim >= 2 else "FID"
    meta.pathname = str(path)
    meta.filename = data_dir.parent
    meta.origin = "agilent"

    # File header parameters
    meta.nblocks = dic.get("nblocks", 1)
    meta.ntraces = dic.get("ntraces", 1)
    meta.np_raw = dic.get("np", 0)
    meta.source_procpar = pdic

    # Procpar parameters — per-dimension lists
    meta.sw = [None] * ndim
    meta.sfrq = [None] * ndim
    meta.dfrq = [None] * ndim
    meta.td = list(data.shape)
    meta.nucleus = [None] * ndim
    meta.encoding = list(reversed(_detect_agilent_encoding(pdic, ndim)))
    meta.iscomplex = [False] * ndim
    meta.isfreq = [False] * ndim

    # Direct dimension (last axis)
    meta.sw[-1] = _extract_procpar_value(pdic, "sw")
    meta.sfrq[-1] = _extract_procpar_value(pdic, "sfrq")
    meta.dfrq[-1] = _extract_procpar_value(pdic, "dfrq")
    nuc = _extract_procpar_string(pdic, "dn") or _extract_procpar_string(pdic, "tn")
    meta.nucleus[-1] = nuc
    np_value = _extract_procpar_integer(pdic, "np")
    meta.iscomplex[-1] = np_value is not None and np_value % 2 == 0 and np_value > 0

    # Spectrometer frequency for direct dim: prefer sfrq, fallback to dfrq
    if meta.sfrq[-1] is None:
        meta.sfrq[-1] = meta.dfrq[-1]

    # Indirect dimensions (first axis(es))
    ni_keys = [
        ("ni", "sw2", "sfrq2", "dfrq2", "dn2", "tn2", "phase"),
        ("ni2", "sw3", "sfrq3", "dfrq3", "dn3", "tn3", "phase2"),
        ("ni3", "sw4", "sfrq4", "dfrq4", "dn4", "tn4", "phase3"),
    ]

    for i, (
        _ni_key,
        sw_key,
        sfrq_key,
        dfrq_key,
        dn_key,
        tn_key,
        phase_key,
    ) in enumerate(ni_keys):
        dim_idx = ndim - 2 - i
        if dim_idx < 0:
            break

        meta.sw[dim_idx] = _extract_procpar_value(pdic, sw_key)
        meta.sfrq[dim_idx] = _extract_procpar_value(pdic, sfrq_key)
        meta.dfrq[dim_idx] = _extract_procpar_value(pdic, dfrq_key)
        if meta.sfrq[dim_idx] is None:
            meta.sfrq[dim_idx] = meta.dfrq[dim_idx]

        nuc = _extract_procpar_string(pdic, dn_key) or _extract_procpar_string(
            pdic, tn_key
        )
        meta.nucleus[dim_idx] = nuc

        # Indirect dimensions are complex if phase array has 2+ values
        phase_vals = _extract_procpar_values(pdic, phase_key)
        meta.iscomplex[dim_idx] = len(phase_vals) >= 2

    # Compute spectral width in Hz (already in Hz for Agilent)
    meta.sw_h = [None] * ndim
    for i in range(ndim):
        if meta.sw[i] is not None:
            meta.sw_h[i] = meta.sw[i] * ur.Hz

    # ns (number of scans)
    meta.ns = _extract_procpar_value(pdic, "ns", 1.0)
    meta.ne = _extract_procpar_value(pdic, "ne", 1.0)

    # Normalise amplitude to ns=1
    ns = float(meta.ns) if meta.ns else 1.0
    ne = float(meta.ne) if meta.ne else 1.0
    fac = ns * ne
    if fac > 0:
        data = data / fac
        meta.ns_norm = 1.0
    else:
        meta.ns_norm = ns

    # Pulse program
    meta.pulprog = _extract_procpar_string(pdic, "array") or _extract_procpar_string(
        pdic, "pulprog"
    )

    # Date
    date_str = _extract_procpar_string(pdic, "date")
    meta.date_raw = date_str

    # Acquisition time
    meta.at = _extract_procpar_value(pdic, "at")

    # Build coordinates
    coords = []

    def _make_coord(axis, size):
        """Build a single coordinate for the given axis."""
        if not meta.isfreq[axis]:
            sw_hz = meta.sw_h[axis]
            if sw_hz is not None and float(sw_hz.magnitude) > 0:
                dw = (1.0 / sw_hz).to("us")
                coordpoints = np.arange(size)
                coord = Coord(coordpoints * dw, title=f"F{axis + 1} acquisition time")
            else:
                coord = Coord(np.arange(size), title=f"F{axis + 1} acquisition time")

            freq = meta.sfrq[axis] or meta.dfrq[axis]
            if freq is not None:
                coord.meta["acquisition_frequency"] = freq * ur.MHz
            return coord

        # Frequency axis
        sw_hz = meta.sw_h[axis]
        freq_mhz = meta.sfrq[axis] or meta.dfrq[axis]

        if sw_hz is not None and freq_mhz is not None and size > 1:
            sizem = max(size - 1, 1)
            deltaf = -float(sw_hz.magnitude) / sizem
            first = float(freq_mhz) * 1e6 - deltaf * sizem / 2.0

            coordpoints = np.arange(size) * deltaf + first
            coord = Coord(coordpoints)
            coord.meta["acquisition_frequency"] = freq_mhz * ur.MHz
            coord.ito("ppm")

            nuc = meta.nucleus[axis]
            if nuc:
                import re

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

    if ndim == 2 and (
        meta.encoding[0] == "STATES" or _has_unvalidated_states_phase(pdic)
    ):
        supported, reason = _validate_agilent_states_profile(pdic, dic, data)
        if not supported:
            raise NotImplementedError(f"Unvalidated Agilent States storage: {reason}")
        channels = _reconstruct_agilent_states(data)
        coords = [
            _make_coord(axis, size) for axis, size in enumerate(channels.shape[1:])
        ]
        meta.acquisition_encoding = tuple(meta.encoding)
        meta.encoding[0] = "UNVALIDATED:AGILENT-STATES"
        meta.normalization_state = "storage-decoded"
        meta.seqfil = _extract_procpar_string(pdic, "seqfil")
        meta.phase = (1, 2)
        meta.ni = _extract_procpar_integer(pdic, "ni")
        meta.arraydim = _extract_procpar_integer(pdic, "arraydim")
        set_channel_data(
            dataset,
            channels,
            coords,
            meta,
            labels=("branch_0", "branch_1"),
            stage="acquisition",
            profile="agilent-phase-1-2-trace-pairs",
        )
    else:
        if ndim == 2 and meta.encoding[0] == "TPPI":
            _require_fft_prevalidation()
            meta.encoding[0] = "UNVALIDATED:AGILENT-TPPI"
        elif (
            ndim == 2
            and "phase" not in pdic
            and _extract_procpar_integer(pdic, "ni", default=0) > 1
        ):
            _require_fft_prevalidation()
            meta.encoding[0] = "UNVALIDATED:AGILENT-F1"
        dataset.data = data
        dataset.meta.update(meta)
        dataset.set_coordset(*tuple(coords))
    return _finalize_agilent_dataset(dataset, data_dir, kwargs)
