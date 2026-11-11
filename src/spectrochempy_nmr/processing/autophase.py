"""Autophase complex 1D NMR spectra and optionally correct their real baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from spectrochempy import Baseline
from spectrochempy import NDDataset
from spectrochempy import ur
from spectrochempy.core.dataset.nddataset import _history_selection_value

if TYPE_CHECKING:
    from nmrglue.process.proc_autophase import PhaseEstimate


@dataclass(frozen=True)
class AutoPhaseResult:
    """Estimate, phased complex spectrum, and optional real baseline outputs."""

    estimate: PhaseEstimate
    phased: NDDataset | None
    baseline: NDDataset | None = None
    corrected: NDDataset | None = None


def autophase(spectrum, *, exclude=None, baseline_options=None, apply=True, **options):
    """Estimate a relative correction on the full grid of a 1D NMR spectrum.

    True samples in the boolean array `exclude` are excluded from estimation
    along with the dataset mask. The dataset mask is retained on the output.
    Estimator options are passed to nmrglue-ng's `estimate_phase`, including
    `method`, `order`, `fixed_p1`, `references` and the provisional `baseline_mask`.

    Pass `apply=False` for diagnostics only. If the estimate fails to converge,
    the result contains diagnostics alone. Pass `baseline_options` to fit a
    separate baseline on the phased real channel using the core `Baseline` processor.
    Use the phased complex intermediate for further phase corrections. Its
    linear phase metadata is relative to this call's input, with the previous
    phase state recorded in history.
    """
    from nmrglue.process import proc_autophase  # noqa: PLC0415

    data = spectrum.data
    coord = spectrum.x if spectrum.ndim == 1 else None
    if (
        spectrum.ndim != 1
        or not np.iscomplexobj(data)
        or coord is None
        or spectrum.meta.get("data_domain") == "time"
        or (coord.units is not None and coord.units.dimensionality == "[time]")
        or spectrum.meta.get("isfreq") == [False]
    ):
        raise ValueError("autophase requires a complex 1D frequency-domain spectrum")
    if "mask" in options:
        raise TypeError("use exclude to mask samples for phase estimation")
    if not apply and baseline_options is not None:
        raise ValueError("baseline correction requires apply=True")
    mask = np.broadcast_to(spectrum.mask, data.shape).copy()
    if exclude is not None:
        exclude = np.asarray(exclude)
        if exclude.shape != data.shape or exclude.dtype != bool:
            raise ValueError("exclude must be a boolean array matching the spectrum")
        mask |= exclude

    estimate = proc_autophase.estimate_phase(data, mask=mask, **options)
    if not apply or not estimate.success:
        return AutoPhaseResult(estimate, None)

    phased = spectrum.copy()
    phased.data = proc_autophase.ps(data, p0=estimate.p0, p1=estimate.p1)
    metadata = phased.meta
    previous_phase = {
        key: [_history_selection_value(value) for value in metadata[key]]
        for key in ("phased", "phc0", "phc1", "pivot", "exptc", "physical_pivots")
        if metadata.get(key) is not None
    }
    readonly = metadata.readonly
    metadata.readonly = False
    try:
        metadata.phased = [True]
        metadata.phc0 = [estimate.p0 * ur.degree]
        metadata.phc1 = [estimate.p1 * ur.degree]
        metadata.pivot = [0]
        metadata.exptc = [0]
        if "physical_pivots" in metadata:
            metadata.physical_pivots = [False]
    finally:
        metadata.readonly = readonly
    parameters = {
        "method": options.get("method", "acme"),
        "order": int(options.get("order", 0)),
        "p0_deg": estimate.p0,
        "p1_deg": estimate.p1,
        "pivot_index": 0,
        "size": data.size,
        "previous_phase_metadata": previous_phase,
    }
    if options.get("fixed_p1") is not None:
        parameters["fixed_p1_deg"] = float(options["fixed_p1"])
    if np.any(mask):
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        parameters["excluded_index_ranges"] = edges.reshape(-1, 2).tolist()
    if len(options.get("references", ())):
        parameters["references"] = [
            [int(value) for value in ref] for ref in options["references"]
        ]
    if options.get("baseline_mask") is not None:
        anchors = np.asarray(options["baseline_mask"]) & ~mask
        edges = np.flatnonzero(np.diff(np.r_[False, anchors, False]))
        parameters["baseline_anchor_index_ranges"] = edges.reshape(-1, 2).tolist()
    phased._append_history_entry(
        operation="nmr.autophase",
        parameters=parameters,
        message="Applied relative automatic NMR phase correction",
    )
    if baseline_options is None:
        return AutoPhaseResult(estimate, phased)

    absorption = phased.real
    processor = Baseline(**baseline_options)
    processor.fit(absorption)
    baseline = processor.baseline
    corrected = absorption.copy()
    corrected.data = absorption.data - baseline.data
    corrected._append_history_entry(
        operation="baseline_correction",
        parameters={
            "model": processor.model,
            "options": {
                key: _history_selection_value(value)
                for key, value in baseline_options.items()
            },
        },
        message=f"Corrected baseline using {processor.model}",
    )
    return AutoPhaseResult(estimate, phased, baseline, corrected)
