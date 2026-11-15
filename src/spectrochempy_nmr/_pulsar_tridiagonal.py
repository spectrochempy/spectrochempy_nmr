"""Optional Numba/LAPACK backend for real single-spin RF Hamiltonians."""

import ctypes
from functools import lru_cache

import numpy as np


def _lp64_dstev(cython_lapack):
    """Return the LP64 capsule only after verifying its pointer signature."""
    capsule = getattr(cython_lapack, "__pyx_capi__", {}).get("dstev")
    if capsule is None or ctypes.sizeof(ctypes.c_int) != np.dtype(np.int32).itemsize:
        return False
    get_name = ctypes.PYFUNCTYPE(ctypes.c_char_p, ctypes.py_object)(
        ("PyCapsule_GetName", ctypes.pythonapi)
    )
    signature = get_name(capsule)
    if (
        not signature
        or not signature.startswith(b"void (")
        or not signature.endswith(b")")
    ):
        return False
    args = signature[6:-1].replace(b" ", b"").split(b",")
    if len(args) != 8 or args[0] != b"char*":
        return False
    integers = (args[1], args[5], args[7])
    doubles = (b"double*", b"__pyx_t_5scipy_6linalg_13cython_lapack_d*")
    return integers == (b"int*",) * 3 and all(args[i] in doubles for i in (2, 3, 4, 6))


@lru_cache(maxsize=1)
def get_tridiagonal_solver():
    """Initialize the optional backend once, or return None for dense fallback."""
    try:
        from numba import njit
        from numba.core.errors import NumbaError
        from numba.extending import get_cython_function_address
        from scipy.linalg import cython_lapack
    except (ImportError, OSError, RuntimeError):
        return None

    try:
        if not _lp64_dstev(cython_lapack):
            return None
        address = get_cython_function_address("scipy.linalg.cython_lapack", "dstev")
        if not address:
            return None
        dstev = ctypes.CFUNCTYPE(None, *([ctypes.c_void_p] * 8))(address)
        return _compile_solver(njit, dstev)
    except (
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        NumbaError,
    ):
        return None


def _compile_solver(njit, dstev):
    # The LAPACK address belongs to this process, so do not cache code on disk.
    @njit
    def batch(offsets, off_diagonal, duration):
        size = off_diagonal.size + 1
        result = np.empty((offsets.shape[0], size, size), dtype=np.complex128)
        diagonal = np.empty(size, dtype=np.float64)
        edge = np.empty(size - 1, dtype=np.float64)
        vectors = np.empty((size, size), dtype=np.float64)
        work = np.empty(2 * size - 2, dtype=np.float64)
        angle_real = np.empty(size, dtype=np.float64)
        angle_imag = np.empty(size, dtype=np.float64)
        dimension = np.array([size], dtype=np.int32)
        info = np.zeros(1, dtype=np.int32)
        job = np.array([ord("V")], dtype=np.uint8)
        if not np.isfinite(duration) or not np.isfinite(off_diagonal).all():
            return None
        for index in range(offsets.shape[0]):
            diagonal[0] = 0
            for j in range(size - 1):
                diagonal[j + 1] = diagonal[j] + offsets[index, j]
                if not np.isfinite(diagonal[j + 1]):
                    return None
                edge[j] = off_diagonal[j]
            dstev(
                job.ctypes.data,
                dimension.ctypes.data,
                diagonal.ctypes.data,
                edge.ctypes.data,
                vectors.ctypes.data,
                dimension.ctypes.data,
                work.ctypes.data,
                info.ctypes.data,
            )
            if info[0] != 0:
                return None
            for j in range(size):
                phase = -2 * np.pi * duration * diagonal[j]
                if not np.isfinite(phase):
                    return None
                angle_real[j] = np.cos(phase)
                angle_imag[j] = np.sin(phase)
            # LAPACK writes eigenvectors as columns, contiguous as rows here.
            for i in range(size):
                for j in range(i, size):
                    real, imag = 0.0, 0.0
                    for k in range(size):
                        scale = vectors[k, i] * vectors[k, j]
                        real += scale * angle_real[k]
                        imag += scale * angle_imag[k]
                    value = complex(real, imag)
                    result[index, i, j] = value
                    result[index, j, i] = value
        return result

    def solve(rf, offsets, duration):
        size = rf.shape[0]
        edge = np.diagonal(rf, 1).copy()
        steps = np.ascontiguousarray(offsets).reshape(-1, size - 1)
        propagators = batch(steps, edge, duration)
        if propagators is None:
            return None
        return propagators.reshape((*offsets.shape[:-1], size, size))

    # Probe the compiled signature before selecting this path for an input pulse.
    probe = solve(np.zeros((2, 2)), np.zeros((1, 1)), 1e-6)
    if probe is None or not np.allclose(probe[0], np.eye(2), rtol=0, atol=1e-14):
        return None
    return solve
