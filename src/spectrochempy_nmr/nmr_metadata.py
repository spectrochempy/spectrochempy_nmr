"""
Canonical NMR metadata extraction.

Centralises all vendor-specific metadata interpretation behind a single
layer.  Each reader plugin provides a ``extract_nmr_metadata`` function
that maps its raw metadata to the canonical :class:`NMRMetadata` object.

The ``Experiment`` class consumes ``NMRMetadata`` and never references
Bruker-specific field names (``nuc1``, ``pulprog``, ``FnMODE``,
``datatype``, etc.) directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Canonical NMR metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NMRMetadata:
    """
    Vendor-neutral NMR metadata.

    All fields use plain Python types (str, int, float, tuple, None)
    so that the dataclass remains free of pint Quantity or any
    vendor-specific object.
    """

    ndim: int
    """Number of data dimensions (from the dataset shape)."""

    domains: tuple[str, ...]
    """Per-dimension domain state.

    Each element is one of ``'time'``, ``'frequency'``, or ``'unknown'``.
    """

    encoding: tuple[str, ...] | None = None
    """Per-dimension quadrature encoding.

    Canonical strings: ``'QF'``, ``'QSIM'``, ``'QSEQ'``, ``'DQD'``,
    ``'TPPI'``, ``'STATES'``, ``'STATES-TPPI'``, ``'ECHO-ANTIECHO'``,
    ``'undefined'``, or ``None``.
    """

    nuclei: tuple[str, ...] | None = None
    """Observed nucleus per dimension (e.g. ``('1H',)``)."""

    pulse_program: str | None = None
    """Pulse program name (vendor-specific string, may be ``None``)."""

    source_kind: str = "unknown"
    """Source data classification.

    One of: ``'fid'``, ``'ser'``, ``'processed_1d'``,
    ``'processed_2d'``, ``'partially_processed'``, ``'unknown'``.
    """

    datatype: str | None = None
    """Reader-reported data type (e.g. ``'FID'``, ``'SER'``, ``'1D'``, ``'2D'``)."""

    iscomplex: tuple[bool, ...] | None = None
    """Whether each dimension carries complex-valued data."""

    spectral_width_hz: tuple[float | None, ...] | None = None
    """Spectral width in Hz per dimension (plain float or None)."""

    spectrometer_freq_mhz: tuple[float | None, ...] | None = None
    """Spectrometer frequency in MHz per dimension (plain float or None)."""

    normalized_interop: bool = False
    """Whether the canonical normalized interop contract supplies this state."""


# ---------------------------------------------------------------------------
# Source-kind inference (shared logic)
# ---------------------------------------------------------------------------


def infer_source_kind(
    ndim: int,
    domains: tuple[str, ...],
    datatype: str | None = None,
) -> str:
    """
    Derive the source-kind string from canonical fields.

    This function is vendor-neutral — it uses only ``ndim``, ``domains``,
    and the reader-reported ``datatype``.
    """
    if ndim == 1:
        if domains == ("time",):
            return "fid"
        if domains == ("frequency",):
            return "processed_1d"
    elif ndim == 2:
        if domains == ("time", "time"):
            return "ser"
        if domains == ("frequency", "frequency"):
            return "processed_2d"
        if domains[0] != domains[1]:
            return "partially_processed"
    return "unknown"


# ---------------------------------------------------------------------------
# Domain summarisation (shared logic)
# ---------------------------------------------------------------------------


def summarise_domain(domains: tuple[str, ...]) -> str:
    """Return a single domain label from per-dimension domain tuple."""
    if all(d == "time" for d in domains):
        return "time"
    if all(d == "frequency" for d in domains):
        return "frequency"
    if any(d == "unknown" for d in domains):
        return "unknown"
    return "mixed"


# ---------------------------------------------------------------------------
# TopSpin / Bruker extraction
# ---------------------------------------------------------------------------

# FnMODE integer → canonical string (Bruker convention).
_FNMODE_TO_CANONICAL: list[str] = [
    "undefined",
    "QF",
    "QSEQ",
    "TPPI",
    "STATES",
    "STATES-TPPI",
    "ECHO-ANTIECHO",
]


def _resolve_encoding(raw_encoding: list) -> tuple[str, ...]:
    """Convert raw encoding list (may contain ints) to canonical strings."""
    resolved = []
    for e in raw_encoding:
        if isinstance(e, int):
            if 0 <= e < len(_FNMODE_TO_CANONICAL):
                resolved.append(_FNMODE_TO_CANONICAL[e])
            else:
                resolved.append(f"unknown({e})")
        else:
            resolved.append(str(e))
    return tuple(resolved)


def _physical_metadata(meta):
    if getattr(meta, "component_axis", None) != "c":
        return meta
    physical = meta.copy()
    physical.readonly = False
    for key, value in list(physical.items()):
        if isinstance(value, list) and len(value) == 3:
            physical[key] = value[1:]
    if physical.encoding is not None:
        physical.encoding = [
            value.removeprefix("PENDING-F2:")
            .removeprefix("DIRECT:")
            .removeprefix("INDIRECT:")
            if isinstance(value, str)
            else value
            for value in physical.encoding
        ]
    physical.ndim = 2
    return physical


def extract_topspin_metadata(meta) -> NMRMetadata:
    """
    Extract :class:`NMRMetadata` from a TopSpin metadata object.

    Parameters
    ----------
    meta : Meta or None
        The ``dataset.meta`` object produced by the TopSpin reader.

    Returns
    -------
    NMRMetadata
        Canonical NMR metadata.  Fields for which the Bruker metadata
        is absent or unrecognisable are set to ``None`` or ``'unknown'``.
    """
    if meta is None or len(meta) == 0:
        return NMRMetadata(ndim=0, domains=())

    meta = _physical_metadata(meta)

    # --- dimensionality ---
    ndim = getattr(meta, "ndim", None)
    if ndim is None or ndim == 0:
        # Bruker reader stores ndim=None; infer from isfreq, iscomplex, etc.
        isfreq = getattr(meta, "isfreq", None)
        iscomplex = getattr(meta, "iscomplex", None)
        encoding = getattr(meta, "encoding", None)
        for candidate in (isfreq, iscomplex, encoding):
            if candidate is not None and len(candidate) > 0:
                ndim = len(candidate)
                break
        else:
            ndim = 0

    # --- domains ---
    isfreq = getattr(meta, "isfreq", None)
    if isfreq is not None:
        domains = tuple("frequency" if f else "time" for f in isfreq)
    else:
        domains = tuple("unknown" for _ in range(ndim))

    # --- encoding ---
    raw_enc = getattr(meta, "encoding", None)
    encoding = _resolve_encoding(raw_enc) if raw_enc is not None else None

    # --- nuclei ---
    raw_nuc = getattr(meta, "nuc1", None)
    nuclei = tuple(raw_nuc) if raw_nuc else None

    # --- pulse program ---
    pulse_program = getattr(meta, "pulprog", None)

    # --- datatype ---
    datatype = getattr(meta, "datatype", None)

    # --- iscomplex ---
    raw_ic = getattr(meta, "iscomplex", None)
    iscomplex = tuple(raw_ic) if raw_ic else None

    # --- spectral width (Hz) ---
    raw_sw = getattr(meta, "sw_h", None)
    sw_hz = None
    if raw_sw is not None:
        sw_hz = tuple(
            float(v.magnitude) if hasattr(v, "magnitude") else (float(v) if v else None)
            for v in raw_sw
        )

    # --- spectrometer frequency (MHz) ---
    raw_sfo = getattr(meta, "sfo1", None)
    sfo_mhz = None
    if raw_sfo is not None:
        sfo_mhz = tuple(
            float(v.magnitude) if hasattr(v, "magnitude") else (float(v) if v else None)
            for v in raw_sfo
        )

    # --- source kind ---
    source_kind = infer_source_kind(ndim, domains, datatype)

    return NMRMetadata(
        ndim=ndim,
        domains=domains,
        encoding=encoding,
        nuclei=nuclei,
        pulse_program=pulse_program,
        source_kind=source_kind,
        datatype=datatype,
        iscomplex=iscomplex,
        spectral_width_hz=sw_hz,
        spectrometer_freq_mhz=sfo_mhz,
    )


# ---------------------------------------------------------------------------
# Agilent / Varian extraction
# ---------------------------------------------------------------------------

_AGILENT_ENCODING_MAP = {
    "QF": "QF",
    "QSIM": "QSIM",
    "QSEQ": "QSEQ",
    "TPPI": "TPPI",
    "STATES": "STATES",
    "STATES-TPPI": "STATES-TPPI",
    "ECHO-ANTIECHO": "ECHO-ANTIECHO",
}


def extract_agilent_metadata(meta) -> NMRMetadata:
    """
    Extract :class:`NMRMetadata` from an Agilent/Varian metadata object.

    Parameters
    ----------
    meta : Meta or None
        The ``dataset.meta`` object produced by the Agilent reader.

    Returns
    -------
    NMRMetadata
        Canonical NMR metadata.
    """
    if meta is None or len(meta) == 0:
        return NMRMetadata(ndim=0, domains=())

    meta = _physical_metadata(meta)

    # --- dimensionality ---
    td = getattr(meta, "td", None)
    ndim = len(td) if td is not None else 0

    # --- domains ---
    isfreq = getattr(meta, "isfreq", None)
    if isfreq is not None:
        domains = tuple("frequency" if f else "time" for f in isfreq)
    else:
        domains = tuple("unknown" for _ in range(ndim))

    # --- encoding ---
    raw_enc = getattr(meta, "encoding", None)
    encoding = None
    if raw_enc is not None:
        encoding = tuple(_AGILENT_ENCODING_MAP.get(str(e), str(e)) for e in raw_enc)

    # --- nuclei ---
    raw_nuc = getattr(meta, "nucleus", None)
    nuclei = None
    if raw_nuc is not None:
        nuclei = tuple(n if n else None for n in raw_nuc)

    # --- pulse program ---
    pulse_program = getattr(meta, "pulprog", None)

    # --- datatype ---
    datatype = getattr(meta, "datatype", None)

    # --- iscomplex ---
    raw_ic = getattr(meta, "iscomplex", None)
    iscomplex = tuple(raw_ic) if raw_ic else None

    # --- spectral width (Hz) ---
    raw_sw = getattr(meta, "sw_h", None)
    sw_hz = None
    if raw_sw is not None:
        sw_hz = tuple(
            float(v.magnitude) if hasattr(v, "magnitude") else (float(v) if v else None)
            for v in raw_sw
        )

    # --- spectrometer frequency (MHz) ---
    raw_sfo = getattr(meta, "sfrq", None)
    sfo_mhz = None
    if raw_sfo is not None:
        sfo_mhz = tuple(
            float(v.magnitude) if hasattr(v, "magnitude") else (float(v) if v else None)
            for v in raw_sfo
        )

    # --- source kind ---
    source_kind = infer_source_kind(ndim, domains, datatype)

    return NMRMetadata(
        ndim=ndim,
        domains=domains,
        encoding=encoding,
        nuclei=nuclei,
        pulse_program=pulse_program,
        source_kind=source_kind,
        datatype=datatype,
        iscomplex=iscomplex,
        spectral_width_hz=sw_hz,
        spectrometer_freq_mhz=sfo_mhz,
    )


# ---------------------------------------------------------------------------
# JEOL extraction
# ---------------------------------------------------------------------------

_JEOL_ENCODING_MAP = {
    "QF": "QF",
    "QSIM": "QSIM",
    "QSEQ": "QSEQ",
    "TPPI": "TPPI",
    "STATES": "STATES",
    "STATES-TPPI": "STATES-TPPI",
    "ECHO-ANTIECHO": "ECHO-ANTIECHO",
}


def extract_jeol_metadata(meta) -> NMRMetadata:
    """
    Extract :class:`NMRMetadata` from a JEOL metadata object.

    Parameters
    ----------
    meta : Meta or None
        The ``dataset.meta`` object produced by the JEOL reader.

    Returns
    -------
    NMRMetadata
        Canonical NMR metadata.
    """
    if meta is None or len(meta) == 0:
        return NMRMetadata(ndim=0, domains=())

    # --- dimensionality ---
    td = getattr(meta, "td", None)
    ndim = len(td) if td is not None else 0

    # --- domains ---
    isfreq = getattr(meta, "isfreq", None)
    if isfreq is not None:
        domains = tuple("frequency" if f else "time" for f in isfreq)
    else:
        domains = tuple("unknown" for _ in range(ndim))

    # --- encoding ---
    raw_enc = getattr(meta, "encoding", None)
    encoding = None
    if raw_enc is not None:
        encoding = tuple(_JEOL_ENCODING_MAP.get(str(e), str(e)) for e in raw_enc)

    # --- nuclei ---
    raw_nuc = getattr(meta, "nucleus", None)
    nuclei = None
    if raw_nuc is not None:
        nuclei = tuple(n if n else None for n in raw_nuc)

    # --- pulse program ---
    pulse_program = getattr(meta, "experiment", None)

    # --- datatype ---
    datatype = getattr(meta, "datatype", None)

    # --- iscomplex ---
    raw_ic = getattr(meta, "iscomplex", None)
    iscomplex = tuple(raw_ic) if raw_ic else None

    # --- spectral width (Hz) ---
    raw_sw = getattr(meta, "sw_h", None)
    sw_hz = None
    if raw_sw is not None:
        sw_hz = tuple(
            float(v.magnitude) if hasattr(v, "magnitude") else (float(v) if v else None)
            for v in raw_sw
        )

    # --- spectrometer frequency (MHz) ---
    raw_sfo = getattr(meta, "sfrq", None)
    sfo_mhz = None
    if raw_sfo is not None:
        sfo_mhz = tuple(
            float(v.magnitude) if hasattr(v, "magnitude") else (float(v) if v else None)
            for v in raw_sfo
        )

    # --- source kind ---
    source_kind = infer_source_kind(ndim, domains, datatype)

    return NMRMetadata(
        ndim=ndim,
        domains=domains,
        encoding=encoding,
        nuclei=nuclei,
        pulse_program=pulse_program,
        source_kind=source_kind,
        datatype=datatype,
        iscomplex=iscomplex,
        spectral_width_hz=sw_hz,
        spectrometer_freq_mhz=sfo_mhz,
    )


# ---------------------------------------------------------------------------
# TecMag extraction
# ---------------------------------------------------------------------------

_TECMAG_ENCODING_MAP = {
    "QF": "QF",
    "QSIM": "QSIM",
    "QSEQ": "QSEQ",
    "TPPI": "TPPI",
    "STATES": "STATES",
    "STATES-TPPI": "STATES-TPPI",
    "ECHO-ANTIECHO": "ECHO-ANTIECHO",
}


def extract_tecmag_metadata(meta) -> NMRMetadata:
    """
    Extract :class:`NMRMetadata` from a TecMag metadata object.

    Parameters
    ----------
    meta : Meta or None
        The ``dataset.meta`` object produced by the TecMag reader.

    Returns
    -------
    NMRMetadata
        Canonical NMR metadata.
    """
    if meta is None or len(meta) == 0:
        return NMRMetadata(ndim=0, domains=())

    # --- dimensionality ---
    td = getattr(meta, "td", None)
    ndim = len(td) if td is not None else 0

    # --- domains ---
    isfreq = getattr(meta, "isfreq", None)
    if isfreq is not None:
        domains = tuple("frequency" if f else "time" for f in isfreq)
    else:
        domains = tuple("unknown" for _ in range(ndim))

    # --- encoding ---
    raw_enc = getattr(meta, "encoding", None)
    encoding = None
    if raw_enc is not None:
        encoding = tuple(_TECMAG_ENCODING_MAP.get(str(e), str(e)) for e in raw_enc)

    # --- nuclei ---
    raw_nuc = getattr(meta, "nucleus", None)
    nuclei = None
    if raw_nuc is not None:
        nuclei = tuple(n if n else None for n in raw_nuc)

    # --- pulse program ---
    pulse_program = getattr(meta, "experiment", None)

    # --- datatype ---
    datatype = getattr(meta, "datatype", None)

    # --- iscomplex ---
    raw_ic = getattr(meta, "iscomplex", None)
    iscomplex = tuple(raw_ic) if raw_ic else None

    # --- spectral width (Hz) ---
    raw_sw = getattr(meta, "sw_h", None)
    sw_hz = None
    if raw_sw is not None:
        sw_hz = tuple(
            float(v.magnitude) if hasattr(v, "magnitude") else (float(v) if v else None)
            for v in raw_sw
        )

    # --- spectrometer frequency (MHz) ---
    raw_sfo = getattr(meta, "sfrq", None)
    sfo_mhz = None
    if raw_sfo is not None:
        sfo_mhz = tuple(
            float(v.magnitude) if hasattr(v, "magnitude") else (float(v) if v else None)
            for v in raw_sfo
        )

    # --- source kind ---
    source_kind = infer_source_kind(ndim, domains, datatype)

    return NMRMetadata(
        ndim=ndim,
        domains=domains,
        encoding=encoding,
        nuclei=nuclei,
        pulse_program=pulse_program,
        source_kind=source_kind,
        datatype=datatype,
        iscomplex=iscomplex,
        spectral_width_hz=sw_hz,
        spectrometer_freq_mhz=sfo_mhz,
    )


# ---------------------------------------------------------------------------
# SIMPSON extraction
# ---------------------------------------------------------------------------

_SIMPSON_ENCODING_MAP = {
    "QF": "QF",
    "QSIM": "QSIM",
    "QSEQ": "QSEQ",
    "TPPI": "TPPI",
    "STATES": "STATES",
    "STATES-TPPI": "STATES-TPPI",
    "ECHO-ANTIECHO": "ECHO-ANTIECHO",
}


def extract_simpson_metadata(meta) -> NMRMetadata:
    """
    Extract :class:`NMRMetadata` from a SIMPSON metadata object.

    Parameters
    ----------
    meta : Meta or None
        The ``dataset.meta`` object produced by the SIMPSON reader.

    Returns
    -------
    NMRMetadata
        Canonical NMR metadata.
    """
    if meta is None or len(meta) == 0:
        return NMRMetadata(ndim=0, domains=())

    # --- dimensionality ---
    td = getattr(meta, "td", None)
    ndim = len(td) if td is not None else 0

    # --- domains ---
    isfreq = getattr(meta, "isfreq", None)
    if isfreq is not None:
        domains = tuple("frequency" if f else "time" for f in isfreq)
    else:
        domains = tuple("unknown" for _ in range(ndim))

    # --- encoding ---
    raw_enc = getattr(meta, "encoding", None)
    encoding = None
    if raw_enc is not None:
        encoding = tuple(_SIMPSON_ENCODING_MAP.get(str(e), str(e)) for e in raw_enc)

    # --- nuclei ---
    raw_nuc = getattr(meta, "nucleus", None)
    nuclei = None
    if raw_nuc is not None:
        nuclei = tuple(n if n else None for n in raw_nuc)

    # --- pulse program ---
    # SIMPSON has no pulse program concept; leave as None
    pulse_program = None

    # --- datatype ---
    datatype = getattr(meta, "datatype", None)

    # --- iscomplex ---
    raw_ic = getattr(meta, "iscomplex", None)
    iscomplex = tuple(raw_ic) if raw_ic else None

    # --- spectral width (Hz) ---
    raw_sw = getattr(meta, "sw_h", None)
    sw_hz = None
    if raw_sw is not None:
        sw_hz = tuple(
            float(v.magnitude) if hasattr(v, "magnitude") else (float(v) if v else None)
            for v in raw_sw
        )

    # --- spectrometer frequency (MHz) ---
    raw_sfo = getattr(meta, "sfrq", None)
    sfo_mhz = None
    if raw_sfo is not None:
        sfo_mhz = tuple(
            float(v.magnitude) if hasattr(v, "magnitude") else (float(v) if v else None)
            for v in raw_sfo
        )

    # --- source kind ---
    source_kind = infer_source_kind(ndim, domains, datatype)

    return NMRMetadata(
        ndim=ndim,
        domains=domains,
        encoding=encoding,
        nuclei=nuclei,
        pulse_program=pulse_program,
        source_kind=source_kind,
        datatype=datatype,
        iscomplex=iscomplex,
        spectral_width_hz=sw_hz,
        spectrometer_freq_mhz=sfo_mhz,
    )


# ---------------------------------------------------------------------------
# Vendor-neutral dispatcher
# ---------------------------------------------------------------------------

_EXTRACTORS = {
    "topspin": extract_topspin_metadata,
    "agilent": extract_agilent_metadata,
    "jeol": extract_jeol_metadata,
    "tecmag": extract_tecmag_metadata,
    "simpson": extract_simpson_metadata,
}


def extract_nmr_metadata(meta, *, ndim=None) -> NMRMetadata:
    """
    Extract :class:`NMRMetadata` from any supported NMR metadata object.

    Normalized interop metadata is consumed structurally before the legacy
    extractors, which dispatch on ``meta.origin``. Origin and raw metadata do
    not influence the normalized path.

    Parameters
    ----------
    meta : Meta or None
        The ``dataset.meta`` object produced by any supported reader.

    Returns
    -------
    NMRMetadata
        Canonical NMR metadata.
    """
    if meta is None or len(meta) == 0:
        return NMRMetadata(ndim=0, domains=())

    if meta.get("interop_dimensions") is not None:
        physical = tuple(meta.get("interop_physical_dimensions", ()))
        dimensions = tuple(meta.get("interop_dimensions", ()))
        acquisition = meta.get("acquisition", {})
        domain = meta.get("data_domain")
        quadrature = meta.get("quadrature_mode")
        states_tppi = quadrature == "states-tppi" and (
            domain,
            meta.get("tppi_alternation_state"),
        ) in (("time", "encoded"), ("mixed", "encoded"), ("frequency", "applied"))
        echo_antiecho = quadrature == "echo-antiecho-stored-2d" and (
            domain,
            meta.get("echo_antiecho_reconstruction"),
        ) in (
            ("time", None),
            ("mixed", None),
            ("frequency", "bruker-rance-kay"),
            ("frequency", "public-branch-fft"),
        )
        if (
            ndim == 3
            and physical == ("F1", "F2")
            and dimensions == ("channels", "F1", "F2")
            and tuple(meta.get("axis_roles", ())) == ("component", "F1", "F2")
            and (states_tppi or echo_antiecho)
            and meta.get("normalization_state") == "storage-normalized"
            and isinstance(acquisition, Mapping)
        ):
            widths = acquisition.get("spectral_width_hz")
            domains = (domain, domain)
            if domain == "mixed":
                domains = tuple(
                    "frequency" if flag else "time" for flag in meta.isfreq[1:]
                )
            return NMRMetadata(
                normalized_interop=True,
                ndim=3,
                domains=domains,
                source_kind=infer_source_kind(2, domains),
                encoding=("STATES-TPPI" if states_tppi else "ECHO-ANTIECHO", "DQD"),
                iscomplex=(True, True),
                spectral_width_hz=tuple(widths)
                if isinstance(widths, (tuple, list))
                else None,
            )
        if (
            ndim == 3
            and physical in (("F1", "F2"), ("F2", "F1"))
            and dimensions == ("channels", *physical)
            and tuple(meta.get("axis_roles", ())) == ("component", *physical)
            and quadrature == "complex-channel-pair-2d"
            and isinstance(acquisition, Mapping)
            and domain == "frequency"
            and meta.get("normalization_state") == "storage-normalized"
        ):
            domains = ("frequency", "frequency")
            display = extract_topspin_metadata(meta)
            return replace(
                display,
                normalized_interop=True,
                domains=domains,
                source_kind=infer_source_kind(2, domains),
                encoding=display.encoding or ("complex", "complex"),
                iscomplex=display.iscomplex or (True, True),
                spectrometer_freq_mhz=display.spectrometer_freq_mhz
                or tuple(acquisition.get("acquisition_frequency_mhz", ()))
                or None,
            )
        if (
            ndim == 2
            and physical in (("F1", "F2"), ("F2", "F1"))
            and dimensions == ("channels", *physical)
            and tuple(meta.get("axis_roles", ())) == physical
            and meta.get("quadrature_mode") == "stored-components-2d"
            and meta.get("normalization_state") == "storage-normalized"
            and isinstance(acquisition, Mapping)
            and acquisition.get("indirect_sampling") == "qf"
            and meta.get("interop_complex_axis") == "F2"
        ):
            domain = meta.get("data_domain")
            flags = meta.get("isfreq")
            if (
                domain == "mixed"
                and isinstance(flags, (tuple, list))
                and len(flags) == 2
            ):
                domains = tuple("frequency" if flag else "time" for flag in flags)
            elif domain in ("time", "frequency"):
                domains = (domain, domain)
            else:
                domains = ()
            if domains:
                widths = acquisition.get("spectral_width_hz")
                return NMRMetadata(
                    normalized_interop=True,
                    ndim=2,
                    domains=domains,
                    source_kind=infer_source_kind(2, domains),
                    encoding=tuple("QF" if dim == "F1" else "DQD" for dim in physical),
                    iscomplex=tuple(dim == "F2" for dim in physical),
                    spectral_width_hz=tuple(widths)
                    if isinstance(widths, (tuple, list))
                    else None,
                )
        if (
            physical != ("F2",)
            or dimensions != ("F2",)
            or meta.get("quadrature_mode") not in ("complex-1d", "real-1d")
        ):
            raise ValueError("unsupported normalized interop structure")
        actual_ndim = 1 if ndim is None else ndim
        if actual_ndim != 1:
            raise NotImplementedError(
                "normalized interop Experiment currently requires one 1D acquisition"
            )
        domain = meta.get("data_domain", "unknown")
        acquisition = meta.get("acquisition", {})
        return NMRMetadata(
            normalized_interop=True,
            ndim=1,
            domains=(domain,),
            source_kind=infer_source_kind(1, (domain,)),
            nuclei=(
                acquisition.get("legacy_presentation", {}).get(
                    "reported_nucleus", acquisition.get("nucleus")
                ),
            ),
            encoding=tuple(meta.get("encoding") or ()) or None,
            pulse_program=meta.get("pulprog"),
            datatype=meta.get("datatype"),
            iscomplex=(meta.get("quadrature_mode") == "complex-1d",),
            spectral_width_hz=(acquisition.get("spectral_width_hz"),),
            spectrometer_freq_mhz=(acquisition.get("observation_frequency_mhz"),),
        )

    origin = getattr(meta, "origin", None)
    extractor = _EXTRACTORS.get(origin)
    if extractor is not None:
        return extractor(meta)

    # Fallback: try TopSpin extraction (most common format)
    return extract_topspin_metadata(meta)
