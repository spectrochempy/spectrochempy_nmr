# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""Shared dataset-level metadata handling for NMR readers."""

from spectrochempy.core.readers.importer import Importer


class SingleAcquisitionImporter(Importer):
    """Keep potentially stacked reads on their compatibility contract."""

    def _switch_protocol(self, key, files, **kwargs):
        kwargs["_nmr_single_acquisition"] = (
            not kwargs.get("merge", True)
            or sum(len(group) for group in self.files.values()) == 1
        )
        return super()._switch_protocol(key, files, **kwargs)


def apply_reader_metadata_options(dataset, kwargs, *, default_origin):
    """Apply documented user overrides without changing vendor metadata."""
    origin = kwargs.get("origin")
    dataset.origin = default_origin if origin is None else origin

    description = kwargs.get("description")
    if description is not None:
        dataset.description = description


__all__ = ["apply_reader_metadata_options"]


def complete_states_metadata(dataset):
    """Complete physical FID descriptors at the reader boundary."""
    from collections.abc import Mapping
    from numbers import Real

    from ..processing.bruker_filter import imported_bruker_filter

    if dataset.ndim != 3 or dataset.meta.get("quadrature_mode") not in (
        "states",
        "states-tppi",
        "echo-antiecho-stored-2d",
        "paired-phase-cycle-2d",
    ):
        return dataset
    meta = dataset.meta
    acquisition = dict(meta.get("acquisition", {}))
    correction = acquisition.get("digital_filter", {})
    if (
        all(
            acquisition.get(key) is not None
            for key in (
                "acquired_shape",
                "nuclei",
                "acquisition_frequency_mhz",
                "reference_frequency_mhz",
            )
        )
        and isinstance(correction, Mapping)
        and correction.get("group_delay_points") is not None
    ):
        return dataset
    source = meta.get("source_metadata") or {}
    source = source if isinstance(source, Mapping) else {}

    def parameters(*names):
        merged = {}
        for name in names:
            values = source.get(name)
            if isinstance(values, Mapping):
                merged.update(values)
        return merged

    axes = (parameters("acqu2", "acqu2s"), parameters("acqu", "acqus"))
    processed = (parameters("proc2", "proc2s"), parameters("proc", "procs"))
    for key, legacy, fields, source_key, units in (
        ("acquisition_frequency_mhz", "sfo1", axes, "SFO1", "MHz"),
        ("reference_frequency_mhz", "sf", processed, "SF", "MHz"),
        ("nuclei", "nuc1", axes, "NUC1", None),
    ):
        if acquisition.get(key) is not None:
            continue
        values = meta.get(legacy)
        values = (
            values[-2:]
            if isinstance(values, (tuple, list)) and len(values) >= 2
            else (None, None)
        )
        resolved = []
        for value, fields_for_axis in zip(values, fields, strict=True):
            if units is not None:
                if hasattr(value, "to"):
                    value = value.to(units).magnitude
                if not isinstance(value, Real) or isinstance(value, bool):
                    value = None
            if value is None:
                value = fields_for_axis.get(source_key)
            resolved.append(value)
        acquisition[key] = tuple(resolved)
    if acquisition.get("acquired_shape") is None:
        counts = meta.get("tdeff")
        acquisition["acquired_shape"] = (
            tuple(counts[-2:])
            if isinstance(counts, (tuple, list)) and len(counts) >= 2
            else tuple(dataset.shape[-2:])
        )
    correction = acquisition.get("digital_filter", {})
    if isinstance(correction, Mapping):
        acquisition = imported_bruker_filter(acquisition, source)
        correction = acquisition.get("digital_filter", {})
        direct = axes[-1]
        if (
            correction.get("group_delay_points") is None
            and direct.get("DECIM") == 1
            and direct.get("GRPDLY") in (None, -1, 0)
        ):
            acquisition["digital_filter"] = {
                **correction,
                "group_delay_points": 0.0,
                "delay_source": "no decimation",
            }
    readonly = meta.readonly
    meta.readonly = False
    meta.acquisition = acquisition
    meta.readonly = readonly
    return dataset
