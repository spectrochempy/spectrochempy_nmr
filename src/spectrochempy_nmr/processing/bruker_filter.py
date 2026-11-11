"""Resolve the Bruker digital-filter metadata at import or replay setup."""

from collections.abc import Mapping
from numbers import Real

import numpy as np


def _delay_from_bruker_metadata(source_metadata):
    """Return the delay in points and its evidence from stored Bruker fields."""
    acqus = source_metadata.get("acqus", {})
    if isinstance(acqus, Mapping):
        delay = acqus.get("GRPDLY")
        if (
            isinstance(delay, Real)
            and not isinstance(delay, (bool, np.bool_))
            and np.isfinite(delay)
            and delay > 0
        ):
            return delay, "GRPDLY"
        firmware, decimation = acqus.get("DSPFVS"), acqus.get("DECIM")
        if (
            all(
                isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                for value in (firmware, decimation)
            )
            and 0 <= firmware < 20
        ):
            from ..readers.read_topspin import bruker_dsp_table  # noqa: PLC0415

            delay = bruker_dsp_table.get(firmware, {}).get(decimation)
            if delay is not None:
                return delay, "DSPFVS/DECIM"
    return None, None


def imported_bruker_filter(acquisition, source_metadata):
    """Complete the normalized filter descriptor without applying a correction."""
    correction = acquisition.get("digital_filter", {})
    if not isinstance(correction, Mapping):
        return acquisition
    correction = dict(correction)
    if correction.get("group_delay_points") is None:
        delay, source = _delay_from_bruker_metadata(source_metadata)
        if delay is None:
            return acquisition
        correction.update(group_delay_points=float(delay), delay_source=source)
    else:
        correction.setdefault("delay_source", "normalized acquisition")
    return {**acquisition, "digital_filter": correction}


def resolve_bruker_delay(dataset, override):
    """Select a delay and preserve the refusal to correct twice."""
    acquisition = dataset.meta.get("acquisition", {})
    operations = dataset.meta.get("applied_operations", ())
    for record in (acquisition, *operations):
        correction = record.get("digital_filter")
        if isinstance(correction, Mapping) and correction.get("applied"):
            raise ValueError("Bruker digital filter has already been corrected")

    if override is not None:
        return override, "caller"

    correction = acquisition.get("digital_filter")
    if (
        isinstance(correction, Mapping)
        and correction.get("group_delay_points") is not None
    ):
        return correction["group_delay_points"], correction.get(
            "delay_source", "normalized acquisition"
        )

    delay, source = _delay_from_bruker_metadata(dataset.meta.get("source_metadata", {}))
    if delay is not None:
        return delay, source
    raise ValueError("Unknown Bruker group delay")
