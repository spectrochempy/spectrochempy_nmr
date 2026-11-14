"""Reconstruct States time-domain data from a Bruker Multiplex SPAM SER."""

from numbers import Integral
from pathlib import Path

import numpy as np

from .. import _check_nmrglue_installation
from .states_fids import states_fids


def _select_order(traces, order):
    """Project paired scans on one harmonic and return the two States FIDs."""
    count = traces.shape[1]
    angle = 2 * np.pi * order * np.arange(count) / count
    cosine, sine = np.cos(angle), np.sin(angle)
    first, second = traces[:, :, 0], traces[:, :, 1]
    difference, total = first - second, first + second
    e1 = np.einsum("p,ipt->it", cosine, difference)
    e1 -= 1j * np.einsum("p,ipt->it", sine, total)
    e2 = np.einsum("p,ipt->it", sine, difference)
    e2 += 1j * np.einsum("p,ipt->it", cosine, total)
    return np.stack((e1, e2)) / count


def read_multiplex_spam(path, *, coherence_order=3, procno=1, allow_partial=False):
    """Read an archived Bruker MQMultiplex32/52 experiment as States FIDs.

    Parameters
    ----------
    path : path-like
        Experiment directory containing the raw SER, or the SER itself.
    coherence_order : int, optional
        Select 3Q (default) or, for a spin of 5/2, 5Q.
    procno : int, optional
        Processing number supplying the direct reference frequency (SF).
    allow_partial : bool, optional
        Ignore a single incomplete phase cycle beyond the declared TD,
        followed by zero padding. Complete extra cycles are rejected.

    Returns
    -------
    NDDataset
        Uncorrected complex E1/E2 time-domain FIDs with axes (c, y, x).
        Amplitudes are divided by the number of excitation phases.
    """
    _check_nmrglue_installation()
    from nmrglue import interop

    if not isinstance(allow_partial, bool):
        raise TypeError("allow_partial must be a boolean")
    if isinstance(procno, bool) or not isinstance(procno, Integral) or procno < 1:
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
    program = direct.get("PULPROG")
    phase_count = {"MQMultiplex32": 7, "MQMultiplex52": 11}.get(program)
    if (
        phase_count is None
        or direct.get("PARMODE") != 2
        or direct.get("AQ_mod") not in (1, 3)
        or direct.get("FnTYPE", 0) != 0
        or "nuslist_raw" in metadata
        or stored.dimensions.get("higher_dimension_files") != ("acqu3s",)
        or metadata["acqu3s"].get("TD") != 2 * phase_count
    ):
        raise ValueError("Expected a uniform MQMultiplex32 or MQMultiplex52 SPAM SER")
    if (
        isinstance(coherence_order, bool)
        or not isinstance(coherence_order, Integral)
        or coherence_order not in ((3,) if phase_count == 7 else (3, 5))
    ):
        raise ValueError(f"Unsupported MQ coherence order for {program}")
    order = int(coherence_order)
    increments = metadata.get("acqu2s", {}).get("TD")
    words = stored.dimensions["declared_td"]
    if not isinstance(increments, Integral) or increments < 2 or words < 4 or words % 2:
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
    branches = _select_order(traces, order)
    del traces
    selection = {
        "coherence_order": order,
        "phase_count": phase_count,
        "spam_pair_order": (0, 180),
        "ignored_traces": extra.shape[0],
        "ignored_nonzero_traces": nonzero.size,
    }
    return states_fids(
        stored,
        branches,
        kind="multiplex_spam",
        selection=selection,
        operation="reconstruct-multiplex-spam",
        scale=1 / phase_count,
        sign="MSM",
        history=f"Reconstructed {order}Q Multiplex SPAM States FIDs",
    )
