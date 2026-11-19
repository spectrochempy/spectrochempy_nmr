"""Compatibility recipe for the common read/QF/magnitude workflow."""

from operator import index
from pathlib import Path


def process_jeol_cosy_magnitude(path, *, f1_size=None, frequency_grid=None):
    """Read and process the historical two-section magnitude recipe.

    Prefer ``read(path, quadrature="qf")`` and ``Experiment`` for new workflows.
    This shortcut explicitly declares the historical QF reconstruction,
    applies squared sine windows, transforms both dimensions and takes the
    minimum at exchanged physical frequencies. It does not infer quadrature
    from the pulse-program name. It is not a phase-sensitive qualification.

    Parameters
    ----------
    path : path-like
        Uniform two-section JDF with the qualified axis arrangement.
    f1_size : int, optional
        F1 FFT length, at least the acquired size. F2 uses the next power of two.
    frequency_grid : pair of array-like, optional
        Output F1/F2 frequencies in Hz; interpolation precedes symmetrization.

    Returns
    -------
    NDDataset
        Real, symmetrized magnitude with frequency coordinates in Hz.
    """
    from ..experiment import Experiment
    from ..readers.jeol_qf import read_jeol_qf

    path = Path(path)
    try:
        raw = read_jeol_qf(path)
    except ValueError as exc:
        raise ValueError(
            f"Expected a supported two-section JEOL COSY JDF: {exc}"
        ) from exc
    n1, n2 = raw.shape
    size1 = n1 if f1_size is None else index(f1_size)
    if size1 < n1:
        raise ValueError("f1_size must be at least the number of acquired F1 points")
    if frequency_grid is not None:
        from .symmetrize import _frequency_axis

        if not isinstance(frequency_grid, (tuple, list)) or len(frequency_grid) != 2:
            raise ValueError("frequency_grid must contain F1 and F2 coordinates in Hz")
        frequency_grid = tuple(_frequency_axis(axis) for axis in frequency_grid)
    weighted = raw.sp(ssb=0, pow=2, dim="x").sp(ssb=0, pow=2, dim="y")
    spectrum = Experiment(weighted).process(size=(size1, 1 << (n2 - 1).bit_length()))
    result = Experiment(abs(spectrum)).symmetrize(frequency_grid=frequency_grid)
    result.name, result.title = path.stem, "COSY magnitude"
    # Retain the historical descriptive marker: this recipe does not establish
    # the physical acquisition mode of the file; QF is explicitly selected.
    result.meta.update(
        {
            "processing_profile": "jeol-cosy-magnitude",
            "physical_f1_mode": "indeterminate",
            "raw_shape": raw.shape,
            "experiment": raw.meta.get("experiment"),
            "source": str(path),
        }
    )
    return result
