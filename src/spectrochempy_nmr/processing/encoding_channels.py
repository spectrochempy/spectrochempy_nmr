"""Explicit array projections for characterized two-branch FFT conventions."""

from __future__ import annotations

import numpy as np


def _extract_components(data):
    data = np.asarray(data)
    if data.ndim < 3 or data.shape[0] != 2 or not np.iscomplexobj(data):
        raise ValueError("encoding requires two complex channels on the first axis")
    return data[0].real, data[0].imag, data[1].real, data[1].imag


def _pack_channels(first, second):
    if np.shape(first) != np.shape(second):
        raise ValueError("channel shapes must agree")
    return np.stack([first, second]).astype(np.result_type(first, second, np.complex64))


def _prepare_states(rr, ri, ir, ii):
    return (rr - 1j * ri) / 2, (ir - 1j * ii) / 2


def _prepare_tppi(rr, ri, ir, ii):
    return rr + 1j * ir, ri + 1j * ii


def _prepare_echoanti(rr, ri, ir, ii):
    return (rr + ir) + 1j * (rr - ir), (ri + ii) - 1j * (ri - ii)
