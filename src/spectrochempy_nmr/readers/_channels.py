"""Native storage for two associated complex branches and their physical axes."""

from __future__ import annotations

import numpy as np

from spectrochempy_nmr.processing.fft_validation import _require_fft_prevalidation


def pad_channel_metadata(meta):
    for key, value in list(meta.items()):
        if isinstance(value, list) and len(value) == 2:
            default = None
            if key == "encoding":
                default = "CHANNELS"
            elif key in ("isfreq", "iscomplex", "phased"):
                default = False
            elif key in ("td", "tdeff", "si"):
                default = 2
            elif key in ("phc0", "phc1", "pivot", "exptc"):
                default = 0
            meta[key] = [default, *value]


def set_channel_data(dataset, data, coords, meta, *, labels, stage, profile):
    """Attach two complex channels without treating their axis as physical."""
    import spectrochempy as scp

    _require_fft_prevalidation()
    data = np.asarray(data)
    if data.ndim != 3 or data.shape[0] != 2 or not np.iscomplexobj(data):
        raise ValueError("channel storage requires a complex (2, nF1, nF2) array")
    if len(coords) != 2 or any(
        coord.size != size for coord, size in zip(coords, data.shape[1:], strict=True)
    ):
        raise ValueError("physical coordinates must match both channel dimensions")
    pad_channel_metadata(meta)
    if stage == "acquisition":
        meta.encoding[-1] = "DIRECT:" + meta.encoding[-1]
    meta.td = list(data.shape)
    meta.component_axis = "c"
    meta.axis_roles = ("component", "F1", "F2")
    meta.component_labels = tuple(labels)
    meta.stage = stage
    meta.storage_profile = profile
    dataset.data = data
    dataset.dims = ["c", "y", "x"]
    dataset.meta.update(meta)
    dataset.set_coordset(scp.Coord.arange(2, dims="c", labels=list(labels)), *coords)


def pair_traces(data):
    """Separate successive traces without changing complex samples or signs."""
    data = np.asarray(data)
    if data.ndim != 2 or not np.iscomplexobj(data) or data.shape[0] % 2:
        raise ValueError("paired traces require a complex 2D array with even row count")
    return np.stack([data[0::2], data[1::2]])
