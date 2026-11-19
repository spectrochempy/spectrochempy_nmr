"""Minimum symmetrization on physical frequency coordinates."""

from numbers import Real

import numpy as np
from scipy.ndimage import map_coordinates
from spectrochempy import Coord, NDDataset, ur


def _frequency_axis(axis):
    values = np.asarray(axis)
    if (
        values.ndim != 1
        or values.size < 2
        or values.dtype.kind not in "iuf"
        or np.ma.is_masked(axis)
    ):
        raise ValueError("Each frequency_grid axis must be a real, unmasked vector")
    values = values.astype(float)
    if not np.isfinite(values).all():
        raise ValueError("Each frequency_grid axis must be finite")
    if not (np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0)):
        raise ValueError("Each frequency_grid axis must be strictly monotonic")
    return values


def _indices(values, axis):
    indices = np.arange(axis.size)
    if axis[0] > axis[-1]:
        axis, indices = axis[::-1], indices[::-1]
    return np.interp(values, axis, indices, left=-1, right=axis.size)


def symmetrize_minimum(dataset, *, frequency_grid=None):
    """Evaluate min(S(F1,F2), S(F2,F1)); zero-fill exchanged points outside the grid."""
    if (
        tuple(dataset.dims) != ("y", "x")
        or tuple(dataset.meta.get("axis_roles", ())) != ("F1", "F2")
        or dataset.meta.get("data_domain") != "frequency"
        or np.iscomplexobj(dataset.data)
        or np.any(dataset.mask)
        or not np.isfinite(dataset.data).all()
        or np.any(dataset.data < 0)
        or dataset.y.units != ur.Hz
        or dataset.x.units != ur.Hz
    ):
        raise ValueError(
            "Minimum symmetrization requires a finite unmasked 2D magnitude in Hz"
        )
    frequencies = dataset.meta.get("acquisition", {}).get("acquisition_frequency_mhz")
    if frequencies is not None and (
        not isinstance(frequencies, (tuple, list))
        or len(frequencies) != 2
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value <= 0
            for value in frequencies
        )
        or not np.isclose(frequencies[0], frequencies[1], rtol=1e-8, atol=0)
    ):
        raise ValueError(
            "Exchanged frequencies require the same observed frequency in F1/F2"
        )
    y, x = (_frequency_axis(axis.data) for axis in (dataset.y, dataset.x))
    if frequency_grid is None:
        out_y, out_x = y, x
    elif isinstance(frequency_grid, (tuple, list)) and len(frequency_grid) == 2:
        out_y, out_x = (_frequency_axis(axis) for axis in frequency_grid)
    else:
        raise ValueError("frequency_grid must contain F1 and F2 coordinates in Hz")

    def sample(a, b):
        indices = np.broadcast_arrays(_indices(a, y), _indices(b, x))
        return map_coordinates(np.asarray(dataset.data, dtype=float), indices, order=1)

    direct = sample(out_y[:, None], out_x)
    exchanged = sample(out_x, out_y[:, None])
    result = NDDataset(np.minimum(direct, exchanged), units=dataset.units)
    result.meta = dataset.meta.copy()
    result.meta.readonly = False
    result.set_coordset(
        y=Coord(out_y, units="Hz", rounding=False, sigdigits=15),
        x=Coord(out_x, units="Hz", rounding=False, sigdigits=15),
    )
    for dim in ("y", "x"):
        result.coordset[dim].meta = dataset.coordset[dim].meta.copy()
    result.meta.update(
        {
            "quadrature_mode": "real-2d",
            "branch_order": ("real",),
            "interop_complex_axis": None,
            "interop_dimensions": ("F1", "F2"),
            "interop_encoding": ("real", "real"),
            "iscomplex": [False, False],
            "isfreq": [True, True],
            "td": list(result.shape),
            "stage": "spectrum",
        }
    )
    result.meta.applied_operations = (
        *result.meta.get("applied_operations", ()),
        {"name": "minimum-symmetrization", "stage": "processing"},
    )
    result.history = "Minimum symmetrization on exchanged physical frequencies"
    return result
