"""Build calibrated States datasets for staged XF2 processing."""

from numbers import Integral, Real
from pathlib import Path

import numpy as np
from nmrglue import interop
from spectrochempy.core.units import ur

from ..interop import from_interop
from ..processing.bruker_filter import imported_bruker_filter


def _positive(value, label):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"Bruker States processing requires a positive {label}")
    return float(value)


def read_states_fids(
    path, *, procno="1", use_tdeff=False, indirect_dwell=None, quadrature=None
):
    """Decode complete States pairs with explicitly selected time calibration."""
    directory = Path(path).resolve().parent
    pdata = directory / "pdata" / str(procno)
    files = [pdata / name for name in ("procs", "proc2s") if (pdata / name).is_file()]
    if pdata / "procs" not in files or (use_tdeff and pdata / "proc2s" not in files):
        raise FileNotFoundError(f"Missing States processing status files in {pdata}")
    stored = interop.read_bruker_ser_storage(directory, procs_files=files)
    direct, indirect = stored.metadata["acqus"], stored.metadata.get("acqu2s", {})
    mode = indirect.get("FnMODE")
    if mode in (None, 0):
        mc2 = stored.metadata.get("proc2s", {}).get("MC2")
        if isinstance(mc2, Integral) and not isinstance(mc2, bool) and 0 <= mc2 <= 5:
            mode = mc2 + 1
    if (
        direct.get("PARMODE") != 1
        or direct.get("FnTYPE", 0) != 0
        or direct.get("AQ_mod") not in (1, 3)
        or isinstance(mode, (bool, np.bool_))
        or (mode is not None and not isinstance(mode, Integral))
        or mode not in ((None, 0, 4) if quadrature == "states" else (4,))
        or "nuslist_raw" in stored.metadata
        or stored.dimensions.get("higher_dimension_files")
    ):
        raise ValueError("Pair selection requires a uniform 2D States acquisition")
    if indirect_dwell == "IN[0]":
        delays = direct.get("IN")
        indirect_dwell = (
            delays[0] if isinstance(delays, (tuple, list)) and delays else None
        )
    elif indirect_dwell is None:
        width = indirect.get("SW_h")
        if width is None:
            width = _positive(indirect.get("SW"), "F1 width in ppm") * _positive(
                indirect.get("SFO1"), "F1 acquisition frequency"
            )
        indirect_dwell = 1 / _positive(width, "F1 width")
    dwell = _positive(indirect_dwell, "indirect dwell in seconds")
    declared = indirect.get("TD")
    words = stored.dimensions["declared_td"]
    count = stored.metadata["proc2s"].get("TDeff") if use_tdeff else declared
    if (
        any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            for value in (declared, count, words)
        )
        or not 4 <= count <= declared <= stored.data.shape[0]
        or count % 2
        or words < 4
        or words % 2
    ):
        raise ValueError(
            "States selection requires complete States pairs within the acquired TD"
        )
    traces = stored.data[:count, :words]
    branches = np.empty((2, count // 2, words // 2), dtype=np.complex128)
    for branch in (0, 1):
        branches[branch].real = traces[branch::2, ::2]
        branches[branch].imag = traces[branch::2, 1::2]
    dataset = states_fids(
        stored,
        branches,
        kind="states_selection",
        selection={
            "retained_traces": count,
            "declared_traces": declared,
            "stored_traces": stored.data.shape[0],
            "used_tdeff": use_tdeff,
        },
        operation="pair-states-traces",
        scale=1.0,
        sign="none",
        history=f"Read {count // 2} complete States pairs",
        indirect_dwell=dwell,
    )
    return dataset


def states_fids(
    stored,
    branches,
    *,
    kind,
    selection,
    operation,
    scale,
    sign,
    history,
    indirect_dwell=None,
):
    metadata = stored.metadata
    direct = metadata["acqus"]
    indirect = metadata.get("acqu2s", {})
    delays = direct.get("IN")
    if (
        indirect_dwell is None
        and isinstance(delays, (tuple, list, np.ndarray))
        and len(delays)
    ):
        indirect_dwell = delays[0]
    dwell = _positive(indirect_dwell, "indirect dwell in seconds")
    widths = (
        _positive(1 / dwell, "F1 width"),
        _positive(direct.get("SW_h"), "F2 width"),
    )
    carrier = _positive(direct.get("SFO1"), "SFO1 carrier")
    reference = _positive(metadata["procs"].get("SF"), "SF reference")
    nucleus = direct.get("NUC1")
    axes = tuple(
        interop.Axis(
            values=np.arange(size) / width,
            unit="s",
            label=nucleus,
            domain="time",
            direction="increasing",
        )
        for size, width in zip(branches.shape[1:], widths, strict=True)
    )
    dimensions = ("channels", "F1", "F2")
    acquisition = imported_bruker_filter(
        {
            "spectral_width_hz": widths,
            "indirect_sampling": "states",
            "dc_offset": {"applied": False},
            "digital_filter": {"applied": False},
            kind: {**selection, "indirect_dwell_seconds": dwell},
        },
        metadata,
    )
    # Use the same stored States convention as the conventional 2D reader.
    result = interop.NormalizedAcquisition(
        data=np.stack(
            (branches[0].real, branches[0].imag, branches[1].real, branches[1].imag)
        ),
        axes=axes,
        dimensions=dimensions,
        physical_dimensions=dimensions[1:],
        domains=("time", "time"),
        quadrature_mode="stored-components-2d",
        branch_order=("rr", "ri", "ir", "ii"),
        channel_axis=0,
        normalization_state="storage-normalized",
        normalization_profile="bruker-ser-states-2d-v1",
        applied_operations=(
            {
                "name": operation,
                "stage": "acquisition",
                "input_dimensions": ("trace", "word"),
                "output_dimensions": dimensions,
                "axis": "F1",
                "scale": scale,
                "conjugation": "none",
                "sign": sign,
                "permutation": None,
                "evidence": direct.get("PULPROG", "Bruker States"),
                **selection,
            },
        ),
        provenance={"source": str(stored.source), "format": stored.format},
        acquisition=acquisition,
        metadata=metadata,
        encoding=("complex", "complex"),
    )
    dataset = from_interop(result, profile="states")

    dataset.meta.update(
        {
            "datatype": "SER",
            "encoding": ["CHANNELS", "PENDING-F2:INDIRECT:STATES", "DIRECT:DQD"],
            "iscomplex": [False, True, True],
            "tdeff": list(dataset.shape),
            "sw_h": [None, *(width * ur.Hz for width in widths)],
            "sfo1": [None, indirect.get("SFO1", carrier) * ur.MHz, carrier * ur.MHz],
            "sf": [
                None,
                metadata.get("proc2s", {}).get("SF", reference) * ur.MHz,
                reference * ur.MHz,
            ],
            "nuc1": [None, indirect.get("NUC1", nucleus), nucleus],
        }
    )
    from ._metadata import complete_states_metadata

    complete_states_metadata(dataset)
    dataset.filename = stored.source
    dataset.history = history
    return dataset
