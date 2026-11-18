"""Select coherences using physical phase-cycle data, independent of storage."""

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
from spectrochempy import Coord, NDDataset
from spectrochempy.core.units import ur


def select_coherence(dataset, *, coherence_order):
    """Project uniform paired excitation phases onto complex States E1/E2 FIDs."""
    meta = dataset.meta
    output_meta = meta.copy()
    output_meta.readonly = False
    acquisition = output_meta.get("acquisition", {})
    cycle = acquisition.get("phase_cycle", {})
    count = cycle.get("phase_count")
    if (
        tuple(dataset.dims) != ("c", "y", "x")
        or meta.get("quadrature_mode") != "paired-phase-cycle-2d"
        or meta.get("normalization_state") != "storage-normalized"
        or meta.get("stage") != "acquisition"
        or meta.get("data_domain") != "time"
        or not isinstance(count, Integral)
        or isinstance(count, (bool, np.bool_))
        or count not in (7, 11)
        or dataset.shape[0] != 2 * count
        or tuple(cycle.get("spam_pair_order", ())) != (0, 180)
        or tuple(acquisition.get("sampling", ())) != ("regular", "regular")
        or any(
            not isinstance(acquisition.get(key), Mapping)
            or acquisition[key].get("applied") is not False
            for key in ("dc_offset", "digital_filter")
        )
    ):
        raise ValueError(
            "select_coherence requires uncorrected paired phase-cycle FIDs"
        )
    if (
        isinstance(coherence_order, (bool, np.bool_))
        or not isinstance(coherence_order, Integral)
        or coherence_order not in ((3,) if count == 7 else (3, 5))
    ):
        raise ValueError("Unsupported MQ coherence order on this excitation phase grid")
    order = int(coherence_order)
    labels = tuple(f"phase{j}:{pair}" for j in range(count) for pair in (0, 180))
    phases = cycle.get("excitation_phases_degrees")
    if (
        dataset.c is None
        or dataset.c.labels is None
        or tuple(dataset.c.labels) != labels
        or not isinstance(phases, (tuple, list))
        or len(phases) != count
        or any(
            isinstance(p, (bool, np.bool_)) or not isinstance(p, Real) for p in phases
        )
        or not np.allclose(phases, 360 * np.arange(count) / count, rtol=1e-12, atol=0)
    ):
        raise ValueError(
            "select_coherence requires the original ordered excitation phases"
        )
    data = np.asarray(dataset.data)
    shape = acquisition.get("acquired_shape")
    widths = acquisition.get("spectral_width_hz")
    if (
        not np.iscomplexobj(data)
        or np.any(dataset.mask)
        or not np.isfinite(data).all()
        or not isinstance(shape, (tuple, list))
        or tuple(shape) != data.shape[1:]
        or any(
            isinstance(n, bool) or not isinstance(n, Integral) or n < 2 for n in shape
        )
        or not isinstance(widths, (tuple, list))
        or len(widths) != 2
    ):
        raise ValueError(
            "select_coherence requires finite unmasked complete acquired FIDs"
        )
    for axis, n, width in zip((dataset.y, dataset.x), shape, widths, strict=True):
        if (
            axis is None
            or axis.units is None
            or not axis.units.is_compatible_with(ur.s)
            or isinstance(width, (bool, np.bool_))
            or not isinstance(width, Real)
            or not np.isfinite(width)
            or width <= 0
            or not np.allclose(
                axis.to("s").data, np.arange(n) / width, rtol=1e-7, atol=0
            )
        ):
            raise ValueError(
                "select_coherence requires the original uniform time grids"
            )
    traces = data.transpose(1, 0, 2).reshape(shape[0], count, 2, shape[1])
    angle = 2 * np.pi * order * np.arange(count) / count
    cosine, sine = np.cos(angle), np.sin(angle)
    first, second = traces[:, :, 0], traces[:, :, 1]
    difference, total = first - second, first + second
    e1 = np.einsum("p,ipt->it", cosine, difference)
    e1 -= 1j * np.einsum("p,ipt->it", sine, total)
    e2 = np.einsum("p,ipt->it", sine, difference)
    e2 += 1j * np.einsum("p,ipt->it", cosine, total)
    result = NDDataset(
        np.stack((e1, e2)) / count,
        dims=("c", "y", "x"),
        coordset=[Coord(labels=["E1", "E2"]), dataset.y.copy(), dataset.x.copy()],
        units=dataset.units,
        name=dataset.name,
    )
    result.meta = output_meta
    result.meta.readonly = False
    selection = {**cycle, "coherence_order": order}
    result.meta.update(
        {
            "quadrature_mode": "states",
            "component_labels": ("E1", "E2"),
            "normalization_profile": "paired-phase-cycle-states-2d-v1",
            "interop_encoding": ("complex", "complex"),
            "encoding": ["CHANNELS", "PENDING-F2:INDIRECT:STATES", "DIRECT:DQD"],
            "iscomplex": [False, True, True],
            "td": list(result.shape),
            "tdeff": list(result.shape),
            "acquisition": {
                **acquisition,
                "indirect_sampling": "states",
                "multiplex_spam": selection,
            },
            "applied_operations": (
                *output_meta.get("applied_operations", ()),
                {
                    "name": "select-coherence",
                    "stage": "processing",
                    "scale": 1 / count,
                    "coherence_order": order,
                    "phase_count": count,
                    "sign": "MSM",
                },
            ),
        }
    )
    result.filename = dataset.filename
    result.history = f"Selected {order}Q coherence as States E1/E2 (per-phase scale)"
    return result
