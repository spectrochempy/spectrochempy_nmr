"""Optional JIT integration of triangular powder densities."""

from functools import lru_cache

import numpy as np


def _deposit_loop(frequencies, weights, triangles, spectrum, origin, step):
    """Accumulate successive differences of the triangular CDF into bins."""
    for triangle in range(triangles.shape[0]):
        i, j, k = triangles[triangle, 0], triangles[triangle, 1], triangles[triangle, 2]
        if min(i, j, k) < 0 or max(i, j, k) >= frequencies.size:
            raise IndexError("PULSAR triangle vertex out of bounds")
        a = (frequencies[i] - origin) / step
        b = (frequencies[j] - origin) / step
        c = (frequencies[k] - origin) / step
        if not (
            -1 < a < spectrum.size and -1 < b < spectrum.size and -1 < c < spectrum.size
        ):
            raise ValueError("PULSAR triangle outside the spectral grid")
        if a > b:
            a, b = b, a
        if b > c:
            b, c = c, b
        if a > b:
            a, b = b, a

        first, last = int(a), int(c)
        mass = weights[i] + weights[j] + weights[k]
        if first == last:
            spectrum[first] += mass
            continue

        previous = 0.0
        ascending = (b - a) * (c - a)
        descending = (c - b) * (c - a)
        for bin_index in range(first, last):
            edge = bin_index + 1
            if edge <= b:
                delta = edge - a
                cumulative = delta * delta / ascending
            else:
                delta = c - edge
                cumulative = 1 - delta * delta / descending
            spectrum[bin_index] += (cumulative - previous) * mass
            previous = cumulative
        spectrum[last] += (1 - previous) * mass


@lru_cache(maxsize=1)
def get_numba_depositor():
    """Compile the deposition kernel once, or return None for NumPy fallback."""
    try:
        from numba import config, njit, types
        from numba.core.errors import NumbaError
    except (AttributeError, ImportError, OSError, RuntimeError):
        return None
    if config.DISABLE_JIT:
        return None

    vector = types.Array(types.float64, 1, "C", readonly=True)
    mesh = types.Array(types.intp, 2, "C", readonly=True)
    signature = types.void(
        vector, vector, mesh, types.float64[::1], types.float64, types.float64
    )
    try:
        kernel = njit(signature)(_deposit_loop)
    except (OSError, RuntimeError, NumbaError):
        return None

    def deposit(axis, frequencies, weights, triangles):
        result = np.zeros_like(axis)
        step = (axis[-1] - axis[0]) / (axis.size - 1)
        kernel(
            np.ascontiguousarray(frequencies),
            np.ascontiguousarray(weights),
            np.ascontiguousarray(triangles, dtype=np.intp),
            result,
            axis[0],
            step,
        )
        return result

    return deposit
