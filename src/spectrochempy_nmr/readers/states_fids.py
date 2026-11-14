"""Build calibrated States datasets for staged XF2 processing."""

from numbers import Real

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


def states_fids(stored, branches, *, kind, selection, operation, scale, sign, history):
    metadata = stored.metadata
    direct = metadata["acqus"]
    delays = direct.get("IN")
    dwell = _positive(
        delays[0]
        if isinstance(delays, (tuple, list, np.ndarray)) and len(delays)
        else None,
        "IN[0] dwell",
    )
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
            "sfo1": [None, None, carrier * ur.MHz],
            "sf": [None, None, reference * ur.MHz],
            "nuc1": [None, nucleus, nucleus],
        }
    )
    dataset.filename = stored.source
    dataset.history = history
    return dataset
