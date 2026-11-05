"""Temporary Agilent-specific adapters for experimental NOESY processing."""

from __future__ import annotations

import numpy as np

from ..processing.complex_channels import process_complex_channels
from ..processing.quadrature import STATES_TPPI


def _meta_value(meta, key, default=None):
    """Return a metadata value without depending on a concrete Meta class."""
    try:
        return meta.get(key, default)
    except AttributeError:
        return default


def _complex_fft(data, axis):
    """Apply the public SpectroChemPy complex FFT to a plain complex array."""
    import spectrochempy as scp  # noqa: PLC0415

    ndim = data.ndim
    dataset = scp.NDDataset(
        data,
        coordset=[scp.Coord.arange(size) for size in data.shape],
    )
    dataset.meta.isfreq = [False] * ndim
    dataset.meta.td = list(data.shape)
    return np.asarray(scp.fft(dataset, dim=axis).data)


def _validate_profile(dataset, ni, np_raw, arraydim, ntraces):
    """Validate the deliberately narrow experimental profile."""
    data = np.asarray(dataset.data)
    if data.ndim != 2 or not np.iscomplexobj(data):
        raise ValueError("Agilent NOESY processing requires a complex 2D array")
    if _meta_value(dataset.meta, "seqfil") != "NOESY":
        raise ValueError("Agilent NOESY processing requires seqfil='NOESY'")
    if ntraces is None:
        raise ValueError("Agilent NOESY processing requires ntraces=1 metadata")
    if isinstance(ntraces, bool) or not isinstance(ntraces, (int, np.integer)):
        raise ValueError("Agilent NOESY ntraces must be an integer")
    if ntraces != 1:
        raise ValueError("Agilent NOESY processing requires ntraces=1")

    phase = _meta_value(dataset.meta, "phase")
    if (
        not isinstance(phase, (tuple, list))
        or len(phase) != 2
        or any(isinstance(value, bool) for value in phase)
        or not all(isinstance(value, (int, np.integer)) for value in phase)
        or tuple(int(value) for value in phase) != (1, 2)
    ):
        raise ValueError("Agilent NOESY processing requires phase=(1, 2)")
    if any(value is None for value in (ni, np_raw, arraydim)):
        raise ValueError("ni, np_raw and arraydim metadata are required")
    if any(isinstance(value, bool) for value in (ni, np_raw, arraydim)):
        raise ValueError("ni, np_raw and arraydim must be integer values")
    if not all(
        isinstance(value, (int, np.integer)) for value in (ni, np_raw, arraydim)
    ):
        raise ValueError("ni, np_raw and arraydim must be integers")
    if ni <= 0 or np_raw <= 0 or np_raw % 2 or arraydim != 2 * ni:
        raise ValueError("inconsistent Agilent NOESY States dimensions")
    expected = (arraydim, np_raw // 2)
    if data.shape != expected:
        raise ValueError(f"data shape {data.shape} does not match {expected}")
    return data


def _has_mask(dataset):
    """Return whether any input point is masked."""
    mask = np.asarray(dataset.mask, dtype=bool)
    return bool(np.any(mask))


def _compute_legacy_noesy_arrays(
    dataset,
    *,
    transport,
    ni,
    np_raw,
    arraydim,
    ntraces,
    reject_masked=False,
):
    """Compute the validated NOESY channel arrays without a storage choice."""
    data = _validate_profile(dataset, ni, np_raw, arraydim, ntraces)
    if reject_masked and _has_mask(dataset):
        raise ValueError("Agilent NOESY FFT processing requires unmasked input")

    if transport == "var2pipe":
        data = ((-1) ** np.arange(data.shape[0]))[:, None] * np.conjugate(data)

    # N * fftshift(ifft(data)) equals conjugate(fftshift(fft(conjugate(data))))
    # for the core FFT convention; do not apply N a second time here.
    f2 = np.conjugate(_complex_fft(np.conjugate(data), axis=1))
    e1 = f2[0::2]
    e2 = f2[1::2]
    c_r = e1.real - 1j * e2.real
    c_i = e1.imag - 1j * e2.imag
    return _complex_fft(c_r, axis=0), _complex_fft(c_i, axis=0), data


def _agilent_channels_adapter(dataset, *, transport, ni, np_raw, arraydim, ntraces):
    """Validate Agilent metadata and produce normalized branch channels."""
    data = _validate_profile(dataset, ni, np_raw, arraydim, ntraces)
    if _has_mask(dataset):
        raise ValueError("Agilent NOESY FFT processing requires unmasked input")
    if transport == "var2pipe":
        data = ((-1) ** np.arange(data.shape[0]))[:, None] * np.conjugate(data)

    branches = np.stack([data[0::2], data[1::2]], axis=0)
    import spectrochempy as scp  # noqa: PLC0415

    channels = scp.NDDataset(
        branches,
        coordset=[
            scp.Coord.arange(2, dims="c", labels=["branch_0", "branch_1"]),
            scp.Coord.arange(ni, dims="y"),
            scp.Coord.arange(np_raw // 2, dims="x"),
        ],
        dims=["c", "y", "x"],
    )
    channels.meta.update(
        {
            "stage": "acquisition",
            "component_axis": "c",
            "component_labels": ("branch_0", "branch_1"),
            "axis_roles": ("component", "F1", "F2"),
            "processing_profile": STATES_TPPI,
            "sequence": "NOESY",
            "vendor": "Agilent",
            "quadrature_mode": STATES_TPPI,
            "branch_order": ("E1", "E2"),
            "f1coef": (1, 0, 0, 0, 0, 0, -1, 0),
            "tppi_alternation_state": "encoded",
            "normalization_state": "normalized",
            "f2_convention": "centered-inverse-unscaled",
            "applied_operations": ("transport", "branch_demultiplex"),
            "transport": transport,
            "coordinates_status": "provisional_index",
            "raw_shape": tuple(data.shape),
            "provenance": _meta_value(dataset.meta, "provenance"),
            "acquisition": _meta_value(dataset.meta, "acquisition"),
            "normalization": _meta_value(dataset.meta, "normalization"),
        }
    )
    channels.meta.isfreq = [False, False, False]
    channels.meta.td = list(channels.shape)
    return channels


def process_agilent_noesy_states_tppi(
    dataset,
    *,
    transport="none",
    ni=None,
    np_raw=None,
    arraydim=None,
    ntraces=None,
    inplace=False,
):
    """Process an explicitly annotated Agilent NOESY States-TPPI dataset.

    This is an experimental, opt-in prototype. The input must be the raw
    complex matrix with ``seqfil='NOESY'`` and ``phase=(1, 2)`` metadata.
    ``transport='none'`` expects NMRPipe-convention samples; the explicit
    ``transport='var2pipe'`` option applies the validated D2473 transport
    operator before the two complex FFTs.

    No quaternion FFT, apodization, phasing, LP, baseline correction or
    canonical physical quadrature interpretation is performed.
    """
    if transport not in ("none", "var2pipe"):
        raise ValueError("transport must be 'none' or 'var2pipe'")
    meta = dataset.meta
    ni = _meta_value(meta, "ni", ni)
    np_raw = _meta_value(meta, "np_raw", _meta_value(meta, "np", np_raw))
    arraydim = _meta_value(meta, "arraydim", arraydim)
    ntraces = _meta_value(meta, "ntraces", ntraces)
    s_r, s_i, data = _compute_legacy_noesy_arrays(
        dataset,
        transport=transport,
        ni=ni,
        np_raw=np_raw,
        arraydim=arraydim,
        ntraces=ntraces,
    )

    rr, ir = s_r.real, s_r.imag
    ri, ii = s_i.real, s_i.imag
    try:
        from spectrochempy_hypercomplex import as_quaternion  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Agilent NOESY processing requires spectrochempy-hypercomplex"
        ) from exc

    quaternion = as_quaternion(rr + 1j * ri, ir + 1j * ii)
    import spectrochempy as scp  # noqa: PLC0415

    result = dataset if inplace else dataset.copy()
    result.data = quaternion
    result.set_coordset(scp.Coord.arange(ni), scp.Coord.arange(np_raw // 2))
    result.meta.update(
        {
            "processing_profile": "agilent-noesy-states-tppi-experimental",
            "transport": transport,
            "fft_dimensions": ("F2", "F1"),
            "states_pairing": "trace[2a]=phase=1; trace[2a+1]=phase=2",
            "mathematical_component_order": ("RR", "RI", "IR", "II"),
            "normalization": "none",
            "apodization": "none",
            "linear_prediction": "none",
            "raw_shape": tuple(data.shape),
            "raw_ni": int(ni),
            "raw_np": int(np_raw),
            "raw_arraydim": int(arraydim),
            "raw_ntraces": int(ntraces),
            "returned_shape": (int(ni), int(np_raw // 2)),
            "data_domain": "frequency",
            "axis_roles": ("F1", "F2"),
            "coordinates_status": "provisional_index",
            "raw_acquisition": {
                "seqfil": "NOESY",
                "phase": (1, 2),
                "ni": int(ni),
                "np": int(np_raw),
                "arraydim": int(arraydim),
                "ntraces": int(ntraces),
            },
        }
    )
    result.meta.td = [int(ni), int(np_raw // 2)]
    result.meta.isfreq = [True, True]
    if inplace:
        return None
    return result


def process_agilent_noesy_states_tppi_channels(
    dataset,
    *,
    transport="none",
    ni=None,
    np_raw=None,
    arraydim=None,
    ntraces=None,
):
    """Return the experimental NOESY spectrum as a complex-channel dataset.

    The result has shape ``(2, ni, np_raw // 2)`` and dimensions ``("c", "y",
    "x")``. Its channels are ``S_R`` and ``S_I``. The input is an explicitly
    annotated, unmasked NOESY profile; no quaternion storage or quaternion FFT
    is used.
    """
    if transport not in ("none", "var2pipe"):
        raise ValueError("transport must be 'none' or 'var2pipe'")
    meta = dataset.meta
    ni = _meta_value(meta, "ni", ni)
    np_raw = _meta_value(meta, "np_raw", _meta_value(meta, "np", np_raw))
    arraydim = _meta_value(meta, "arraydim", arraydim)
    ntraces = _meta_value(meta, "ntraces", ntraces)
    channels = _agilent_channels_adapter(
        dataset,
        transport=transport,
        ni=ni,
        np_raw=np_raw,
        arraydim=arraydim,
        ntraces=ntraces,
    )
    result = process_complex_channels(
        channels,
        reconstruction=STATES_TPPI,
    )
    result.meta.update(
        {
            "processing_profile": "states-tppi-experimental",
            "stage": "spectrum",
            "transport": transport,
            "component_axis": "c",
            "component_labels": ("S_R", "S_I"),
            "axis_roles": ("component", "F1", "F2"),
            "mathematical_component_order": ("RR", "RI", "IR", "II"),
            "data_domain": "frequency",
            "coordinates_status": "provisional_index",
            "normalization": _meta_value(result.meta, "normalization", "none"),
            "provenance": _meta_value(meta, "provenance"),
            "raw_shape": tuple(dataset.data.shape),
            "raw_ni": int(ni),
            "raw_np": int(np_raw),
            "raw_arraydim": int(arraydim),
            "raw_ntraces": int(ntraces),
            "returned_shape": (2, int(ni), int(np_raw // 2)),
            "raw_acquisition": {
                "seqfil": "NOESY",
                "phase": (1, 2),
                "ni": int(ni),
                "np": int(np_raw),
                "arraydim": int(arraydim),
                "ntraces": int(ntraces),
            },
        }
    )
    result.meta.td = [2, int(ni), int(np_raw // 2)]
    result.meta.isfreq = [False, True, True]
    return result
