"""NMRPipe-convention FFTs for explicitly imported States-TPPI channels."""

import numpy as np

from .qf_interop import _fft_stage


def _cross_pairs(f2):
    cross = np.empty_like(f2)
    cross[0].real, cross[0].imag = f2.real
    cross[1].real, cross[1].imag = f2.imag
    return cross


def process_states_tppi_fft(dataset, widths):
    import spectrochempy as scp  # noqa: PLC0415

    data = np.asarray(dataset.data)
    if (
        tuple(dataset.dims) != ("c", "y", "x")
        or data.shape[0] != 2
        or dataset.c is None
        or dataset.c.labels is None
        or tuple(dataset.c.labels) != ("E1", "E2")
        or not np.iscomplexobj(data)
        or np.any(dataset.mask)
        or not np.isfinite(data).all()
    ):
        raise ValueError(
            "States-TPPI FFT requires the imported unmasked E1/E2 channels"
        )
    if widths is None or len(widths) != 2:
        raise ValueError("States-TPPI FFT requires two spectral widths")

    coordinates = {}
    for dim, size, width in zip(("y", "x"), data.shape[1:], widths, strict=True):
        axis = dataset.coordset[dim]
        if (
            size < 2
            or axis.units != scp.ur.s
            or isinstance(width, (bool, np.bool_))
            or not isinstance(width, (int, float, np.integer, np.floating))
            or not np.isfinite(width)
            or width <= 0
            or not np.allclose(axis.data, np.arange(size) / width, rtol=1e-9, atol=0)
        ):
            raise ValueError("States-TPPI FFT requires the declared uniform time grids")
        hz = (size // 2 - np.arange(size)) * (width / size)
        coordinates[dim] = scp.Coord(
            hz, units="Hz", title=axis.title, rounding=False, sigdigits=15
        )

    f2 = _fft_stage(data, -1, direct=True)
    cross = _cross_pairs(f2)
    cross[:, 1::2] *= -1  # NMRPipe FT -alt.
    f1 = _fft_stage(cross, -2, direct=True)
    planes = np.stack((f1[0].real, f1[1].real, f1[0].imag, f1[1].imag))
    labels = ("rr", "ri", "ir", "ii")
    result = scp.NDDataset(planes, dtype=planes.dtype, dims=["c", "y", "x"])
    result.set_coordset(c=scp.Coord(labels=list(labels)), **coordinates)
    result.meta.update(dict(dataset.meta.copy()))
    result.meta.data_domain = "frequency"
    result.meta.stage = "spectrum"
    result.meta.component_labels = labels
    result.meta.isfreq = [False, True, True]
    result.meta.td = list(planes.shape)
    result.meta.tppi_alternation_state = "applied"
    return result
