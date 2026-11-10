"""Internal linear-prediction helpers for NMR time-domain traces."""

from __future__ import annotations

import numpy as np


def _linear_predict_after(data, final_size, order):
    """
    Extend time-domain traces along their last axis with forward LP.

    Each trace is fitted independently by a least-squares autoregressive model
    of ``order`` points. Roots outside the unit circle are projected onto it to
    prevent an increasing extrapolation.
    """
    values = np.asarray(data)
    if values.ndim == 0:
        msg = "data must have at least one dimension"
        raise ValueError(msg)
    if values.dtype.kind not in "fc":
        msg = "data must have a floating-point or complex dtype"
        raise TypeError(msg)
    if not np.isfinite(values).all():
        msg = "data must contain only finite values"
        raise ValueError(msg)
    initial_size = values.shape[-1]
    if final_size < initial_size:
        msg = "final_size must be at least the input size"
        raise ValueError(msg)
    if not 0 < order <= initial_size // 2:
        msg = "order must be positive and at most half the input size"
        raise ValueError(msg)

    output = np.empty((*values.shape[:-1], final_size), dtype=values.dtype)
    output[..., :initial_size] = values
    if final_size == initial_size:
        return output

    for trace in output.reshape(-1, final_size):
        coefficients = _forward_coefficients(trace[:initial_size], order)
        for index in range(initial_size, final_size):
            trace[index] = np.dot(trace[index - order : index], coefficients)
    return output


def _forward_coefficients(trace, order):
    """Fit and stabilize the forward autoregressive coefficients for one trace."""
    rows = trace.size - order
    matrix = np.empty((rows, order), dtype=trace.dtype)
    for index in range(rows):
        matrix[index] = trace[index : index + order]
    coefficients = np.linalg.lstsq(matrix, trace[order:], rcond=None)[0]

    polynomial = np.empty(order + 1, dtype=coefficients.dtype)
    polynomial[0] = 1.0
    polynomial[1:] = -coefficients[::-1]
    roots = np.roots(polynomial)
    roots[np.abs(roots) > 1.0] /= np.abs(roots[np.abs(roots) > 1.0])
    coefficients = -np.poly(roots)[:0:-1]
    if np.isrealobj(trace):
        coefficients = np.real_if_close(coefficients)
    return coefficients


def _burg_predict_after(data, final_size, order):
    """Extend complex traces along the last axis using Burg reflection coefficients."""
    size = data.shape[-1]
    scale = np.max(np.abs(data), axis=-1, keepdims=True)
    normalized = np.divide(data, scale, out=np.zeros_like(data), where=scale > 0)
    forward, backward = normalized[..., 1:], normalized[..., :-1]
    coefficients = np.ones((*data.shape[:-1], 1), dtype=data.dtype)
    for _ in range(order):
        energy = np.sum(np.abs(forward) ** 2 + np.abs(backward) ** 2, axis=-1)
        cross = -2 * np.sum(backward.conj() * forward, axis=-1)
        reflection = np.divide(
            cross, energy, out=np.zeros_like(cross), where=energy > 0
        )[..., None]
        updated = np.empty((*data.shape[:-1], coefficients.shape[-1] + 1), data.dtype)
        updated[..., 0] = 1
        updated[..., -1] = reflection[..., 0]
        updated[..., 1:-1] = (
            coefficients[..., 1:] + reflection * coefficients[..., :0:-1].conj()
        )
        coefficients = updated
        forward, backward = (
            (forward + reflection * backward)[..., 1:],
            (backward + reflection.conj() * forward)[..., :-1],
        )

    output = np.empty((*data.shape[:-1], final_size), dtype=data.dtype)
    output[..., :size] = data
    predictor = -coefficients[..., :0:-1]
    for index in range(size, final_size):
        output[..., index] = np.sum(
            output[..., index - order : index] * predictor, axis=-1
        )
    return output
