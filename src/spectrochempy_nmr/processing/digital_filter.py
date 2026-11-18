"""Time-domain delay correction and normalized delay metadata."""

from collections.abc import Mapping

import numpy as np


def correct_delay_time(data, phase):
    """Apply the qualified integer shift and boundary treatment."""
    phase = np.floor(phase)
    si = data.shape[-1]
    s = float(si)

    pdata = np.fft.fft(np.fft.ifftshift(data, -1), axis=-1) / s
    pdata = pdata * np.exp(2.0j * np.pi * phase * np.arange(s) / s)
    pdata = np.fft.fftshift(np.fft.ifft(pdata, axis=-1), -1) * s

    skip = int(np.floor(phase + 2.0))
    add = int(max(skip - 6, 0))

    if add > 0:
        pdata[..., :add] = pdata[..., :add] + pdata[..., : -(add + 1) : -1]

    return pdata[..., :-skip]


def resolve_group_delay(dataset, override):
    acquisition = dataset.meta.get("acquisition", {})
    operations = dataset.meta.get("applied_operations", ())
    for record in (acquisition, *operations):
        correction = record.get("digital_filter")
        if isinstance(correction, Mapping) and correction.get("applied"):
            raise ValueError("Digital filter has already been corrected")
    if override is not None:
        return override, "caller"
    correction = acquisition.get("digital_filter", {})
    if (
        isinstance(correction, Mapping)
        and correction.get("group_delay_points") is not None
    ):
        return correction["group_delay_points"], correction.get(
            "delay_source", "normalized acquisition"
        )
    raise ValueError("Unknown group delay in normalized acquisition")
