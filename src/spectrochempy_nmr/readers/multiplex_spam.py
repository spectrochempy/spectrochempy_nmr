"""Decode paired phase-cycle storage without selecting a coherence."""

from numbers import Integral
from pathlib import Path

import numpy as np
from spectrochempy import Coord, NDDataset
from spectrochempy.core.units import ur

from .. import _check_nmrglue_installation
from ..interop import _plain
from ..processing.bruker_filter import imported_bruker_filter
from ._metadata import complete_states_metadata
from .states_fids import fid_calibration


def read_paired_fids(path, *, procno=1, allow_partial=False, indirect_dwell=None):
    """Decode uniform excitation phases with adjacent 0/180 SPAM pairs."""
    _check_nmrglue_installation()
    from nmrglue import interop

    if not isinstance(allow_partial, bool):
        raise TypeError("allow_partial must be a boolean")
    if not str(procno).isdigit() or int(procno) < 1:
        raise ValueError("procno must be a positive integer")
    directory = Path(path).resolve()
    if directory.name == "ser":
        directory = directory.parent
    procs = directory / "pdata" / str(procno) / "procs"
    if not procs.is_file():
        raise FileNotFoundError(f"Multiplex SPAM requires {procs} for the F2 reference")

    stored = interop.read_bruker_ser_storage(directory, procs_files=[procs])
    metadata = stored.metadata
    direct = metadata.get("acqus", {})
    cycle_td = metadata.get("acqu3s", {}).get("TD")
    phase_count = cycle_td // 2 if isinstance(cycle_td, Integral) else None
    if (
        phase_count not in (7, 11)
        or direct.get("PARMODE") != 2
        or direct.get("AQ_mod") not in (1, 3)
        or direct.get("FnTYPE", 0) != 0
        or "nuslist_raw" in metadata
        or stored.dimensions.get("higher_dimension_files") != ("acqu3s",)
        or cycle_td != 2 * phase_count
    ):
        raise ValueError(
            "Expected a uniform paired phase cycle with 7 or 11 excitation phases"
        )
    increments = metadata.get("acqu2s", {}).get("TD")
    words = stored.dimensions["declared_td"]
    if (
        isinstance(increments, (bool, np.bool_))
        or not isinstance(increments, Integral)
        or increments < 2
        or words < 4
        or words % 2
    ):
        raise ValueError(
            "Multiplex SPAM requires complete phase cycles and complex FIDs"
        )
    expected = increments * 2 * phase_count
    if stored.data.shape[0] < expected:
        raise ValueError("SER is shorter than the declared Multiplex SPAM acquisition")
    extra = stored.data[expected:, :words]
    nonzero = np.flatnonzero(np.any(extra, axis=1))
    if nonzero.size and (
        not allow_partial
        or nonzero[-1] >= 2 * phase_count - 1
        or not np.array_equal(nonzero, np.arange(nonzero.size))
    ):
        raise ValueError(
            "Unexpected extra SER data. Set allow_partial=True only for a single "
            "incomplete phase cycle after the declared increments."
        )
    pairs = stored.data[:expected, :words].reshape(increments, phase_count, 2, words)
    traces = np.empty((*pairs.shape[:-1], words // 2), dtype=np.complex128)
    traces.real = pairs[..., ::2]
    traces.imag = pairs[..., 1::2]
    if not np.isfinite(traces).all():
        raise ValueError("Multiplex SPAM requires finite acquired samples")
    if indirect_dwell == "IN[0]":
        indirect_dwell = None
    calibration = fid_calibration(metadata, (increments, words // 2), indirect_dwell)
    widths = calibration["spectral_width_hz"]
    labels = tuple(f"phase{j}:{pair}" for j in range(phase_count) for pair in (0, 180))
    descriptor = {
        "phase_count": phase_count,
        "excitation_phases_degrees": tuple(360 * np.arange(phase_count) / phase_count),
        "spam_pair_order": (0, 180),
        "ignored_traces": extra.shape[0],
        "ignored_nonzero_traces": int(nonzero.size),
        "indirect_dwell_seconds": 1 / widths[0],
    }
    acquisition = imported_bruker_filter(
        {
            **calibration,
            "sampling": ("regular", "regular"),
            "phase_cycle": descriptor,
            "dc_offset": {"applied": False},
            "digital_filter": {"applied": False},
        },
        metadata,
    )
    dataset = NDDataset(
        traces.reshape(increments, 2 * phase_count, -1).transpose(1, 0, 2),
        dims=("c", "y", "x"),
        coordset=[
            Coord(labels=list(labels)),
            Coord(np.arange(increments) / widths[0], units="s", rounding=False),
            Coord(np.arange(words // 2) / widths[1], units="s", rounding=False),
        ],
        units="a.u.",
    )
    dataset.meta.update(
        {
            "acquisition": acquisition,
            "source_metadata": _plain(metadata),
            "provenance": {"source": str(stored.source), "format": stored.format},
            "interop_dimensions": ("channels", "F1", "F2"),
            "interop_physical_dimensions": ("F1", "F2"),
            "interop_domains": ("time", "time"),
            "interop_encoding": ("phase-cycle", "complex"),
            "axis_roles": ("component", "F1", "F2"),
            "quadrature_mode": "paired-phase-cycle-2d",
            "component_axis": "c",
            "component_labels": labels,
            "normalization_state": "storage-normalized",
            "normalization_profile": "paired-phase-cycle-2d-v1",
            "data_domain": "time",
            "stage": "acquisition",
            "datatype": "SER",
            "td": list(dataset.shape),
            "tdeff": list(dataset.shape),
            "isfreq": [False, False, False],
            "iscomplex": [False, False, True],
            "sw_h": [None, *(width * ur.Hz for width in widths)],
            "sfo1": [
                None,
                *(v * ur.MHz for v in calibration["acquisition_frequency_mhz"]),
            ],
            "sf": [None, *(v * ur.MHz for v in calibration["reference_frequency_mhz"])],
            "nuc1": [None, *calibration["nuclei"]],
            "applied_operations": (
                {"name": "decode-paired-phase-cycle", "stage": "acquisition"},
            ),
        }
    )
    complete_states_metadata(dataset)
    dataset.filename = stored.source
    dataset.history = "Read paired excitation-phase scans without coherence selection"
    return dataset


def read_multiplex_spam(path, *, coherence_order=3, procno=1, allow_partial=False):
    """Compatibility shortcut for ``read(..., phase_cycle='paired')`` and selection.

    Parameters
    ----------
    path : path-like
        Experiment directory or raw SER.
    coherence_order : int
        Select 3Q or 5Q (5Q requires eleven excitation phases).
    procno : int
        Processing number supplying the F2 reference frequency.
    allow_partial : bool
        Explicitly discard one incomplete cycle after the declared increments.

    Returns
    -------
    NDDataset
        Uncorrected complex States E1/E2 FIDs, normalized per excitation phase.
    """
    from ..processing.phase_cycle import select_coherence

    if isinstance(procno, bool) or not isinstance(procno, Integral) or procno < 1:
        raise ValueError("procno must be a positive integer")
    return select_coherence(
        read_paired_fids(path, procno=procno, allow_partial=allow_partial),
        coherence_order=coherence_order,
    )
