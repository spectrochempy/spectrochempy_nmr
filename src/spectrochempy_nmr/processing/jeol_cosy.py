"""Explicit magnitude processing for uniform JEOL proton COSY acquisitions."""

from operator import index
from pathlib import Path

from nmrglue import jeol
import numpy as np
from scipy.fft import fft2
from scipy.fft import fftshift
from scipy.ndimage import map_coordinates
from spectrochempy import Coord
from spectrochempy import NDDataset


def process_jeol_cosy_magnitude(path, *, f1_size=None, frequency_grid=None):
    """Reconstruct a real COSY magnitude spectrum from a single raw JEOL JDF.

    Parameters
    ----------
    path : str or os.PathLike
        Local JDF of a uniform proton COSY acquired with ``cosy.jxp`` or
        ``cosy_pfg.jxp`` in the supported two-section layout.
    f1_size : int, optional
        F1 FFT length, at least the acquired size. The default uses the
        acquired size. F2 is zero-filled to the next power of two.
    frequency_grid : tuple of array-like, optional
        F1 and F2 output coordinates in Hz. Supply finite, unmasked real
        vectors in strictly monotonic order. The default is the FFT grid.
        Supply an export grid to evaluate the symmetry directly on that grid.

    Returns
    -------
    NDDataset
        Real spectrum with frequency axes in Hz in F1, F2 order. The
        symmetric minimum is evaluated at exchanged physical frequencies.
        Points whose exchanged coordinates lie outside the grid are zero.

    Notes
    -----
    This experimental recipe uses squared sine windows on the acquired
    points, a double FFT, magnitude and minimum symmetrization. The physical
    F1 quadrature mode is recorded as indeterminate.
    """
    path = Path(path)
    metadata, fid = jeol.read(path)
    header = metadata["header"]
    params = metadata["parameters"]
    experiment = params.get("experiment")
    if (
        experiment not in ("cosy.jxp", "cosy_pfg.jxp")
        or header["data_format"] != "two_d"
        or header["data_axis_type"][:2] != ["real_complex", "real_complex"]
        or header["reversed"][:2] != [False, True]
        or header["data_axis_ranged"][:2] != ["ranged", "ranged"]
        or fid.ndim != 2
        or not np.iscomplexobj(fid)
    ):
        raise ValueError("Expected a supported two-section JEOL proton COSY JDF")

    n1, n2 = fid.shape
    points = [n2, n1]
    if (
        min(points) < 2
        or [params.get("x_points"), params.get("y_points")] != points
        or header["data_points"][:2] != points
        or header["data_offset_start"][:2] != [0, 0]
        or header["data_offset_stop"][:2] != [n2 - 1, n1 - 1]
        or any(unit[-1] != "second" for unit in header["units"][:2])
        or any(params.get(f"{axis}_domain") not in ("1H", "Proton") for axis in "xy")
    ):
        raise ValueError("Expected a complete, uniformly sampled JEOL proton COSY FID")

    try:
        widths = np.asarray([params["y_sweep"], params["x_sweep"]], dtype=float)
        frequencies = np.asarray([params["y_freq"], params["x_freq"]], dtype=float)
        offsets = np.asarray([params["y_offset"], params["x_offset"]], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Missing or invalid JEOL COSY frequency calibration") from exc
    if (
        not np.isfinite([widths, frequencies, offsets]).all()
        or (widths <= 0).any()
        or (frequencies <= 0).any()
        or not np.isclose(frequencies[0], frequencies[1], rtol=1e-8, atol=0)
    ):
        raise ValueError("Invalid JEOL proton COSY frequency calibration")
    stop = (np.asarray(fid.shape) - 1) / widths
    if header["data_axis_start"][:2] != [0.0, 0.0] or not np.allclose(
        header["data_axis_stop"][:2], stop[::-1], rtol=1e-5, atol=0
    ):
        raise ValueError("JEOL COSY time coordinates disagree with uniform sampling")

    size1 = n1 if f1_size is None else index(f1_size)
    if size1 < n1:
        raise ValueError("f1_size must be at least the number of acquired F1 points")
    if frequency_grid is not None:
        output1, output2 = (_frequency_axis(axis) for axis in frequency_grid)

    size2 = 1 << (n2 - 1).bit_length()
    window1 = np.sin(np.pi * np.arange(n1) / (n1 - 1)) ** 2
    window2 = np.sin(np.pi * np.arange(n2) / (n2 - 1)) ** 2
    magnitude = abs(fftshift(fft2(fid * window1[:, None] * window2, s=(size1, size2))))

    # Convert FFT indices to the qualified JEOL COSY orientation.
    magnitude = magnitude[(2 * (size1 // 2) - np.arange(size1)) % size1][
        :, (-np.arange(size2)) % size2
    ]
    carrier1, carrier2 = offsets * frequencies / 1e6
    sw1, sw2 = widths
    step1, step2 = sw1 / size1, sw2 / size2
    f1 = carrier1 + (np.arange(size1) - size1 // 2) * step1
    f2 = carrier2 + sw2 / 2 - sw2 / (2 * n2) - np.arange(size2) * step2
    if frequency_grid is None:
        output1, output2 = f1[::-1], f2
        direct = magnitude[::-1]
    else:
        direct = _sample_magnitude(magnitude, output1[:, None], output2, f1, f2)

    exchanged = _sample_magnitude(magnitude, output2, output1[:, None], f1, f2)
    coord_options = {"linearize_below": 0.0, "rounding": False, "sigdigits": 15}
    result = NDDataset(
        np.minimum(direct, exchanged),
        name=path.stem,
        title="COSY magnitude",
    )
    result.set_coordset(
        y=Coord(output1, units="Hz", title="F1", **coord_options),
        x=Coord(output2, units="Hz", title="F2", **coord_options),
    )
    result.meta.update(
        {
            "processing_profile": "jeol-cosy-magnitude",
            "physical_f1_mode": "indeterminate",
            "axis_roles": ("F1", "F2"),
            "isfreq": [True, True],
            "raw_shape": fid.shape,
            "experiment": experiment,
            "source": str(path),
        }
    )
    return result


def _frequency_axis(axis):
    """Validate coordinates before converting them to floating point."""
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
    if not ((values[1:] > values[:-1]).all() or (values[1:] < values[:-1]).all()):
        raise ValueError("Each frequency_grid axis must be strictly monotonic")
    return values


def _sample_magnitude(magnitude, y, x, f1, f2):
    """Interpolate the unsymmetrized FFT on a grid of physical frequencies."""
    coordinates = np.broadcast_arrays(
        np.interp(y, f1, np.arange(f1.size), left=-1, right=f1.size),
        np.interp(x, f2[::-1], np.arange(f2.size - 1, -1, -1), left=f2.size, right=-1),
    )
    return map_coordinates(magnitude, coordinates, order=1)
