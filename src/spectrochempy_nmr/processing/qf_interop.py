"""FFT-only processing for an explicitly normalized QF acquisition."""

import numpy as np


def _fft_stage(values, axis, *, direct):
    if direct:
        # NMRPipe FT -nodf uses the unnormalized positive-sign transform.
        transformed = np.fft.ifft(values, axis=axis) * values.shape[axis]
    else:
        # The qualified QF indirect step uses NMRPipe FT -neg.
        transformed = np.fft.fft(values, axis=axis).conj()
    return np.fft.fftshift(transformed, axes=axis)


def process_qf_fft(dataset, widths):
    import spectrochempy as scp  # noqa: PLC0415

    if tuple(dataset.dims) != ("y", "x"):
        raise ValueError(
            "normalized 2D QF FFT requires the imported dataset dimension order"
        )
    data = np.asarray(dataset.data)
    if not np.iscomplexobj(data) or np.any(dataset.mask) or not np.isfinite(data).all():
        raise ValueError(
            "normalized 2D QF FFT requires finite unmasked complex samples"
        )
    if widths is None or len(widths) != 2:
        raise ValueError("normalized 2D QF FFT requires two spectral widths")

    coordinates = {}
    for dim, size, width in zip(dataset.dims, data.shape, widths, strict=True):
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
            raise ValueError(
                "normalized 2D QF FFT requires the declared uniform time grids"
            )
        # Carrier-relative NMRPipe grid; the acquisition does not establish a chemical zero.
        hz = (size // 2 - np.arange(size)) * (width / size)
        coordinates[dim] = scp.Coord(
            hz, units="Hz", title=axis.title, rounding=False, sigdigits=15
        )

    f2 = tuple(dataset.meta.axis_roles).index("F2")
    spectrum = _fft_stage(_fft_stage(data, f2, direct=True), 1 - f2, direct=False)
    result = dataset.copy()
    result.data = spectrum
    result.set_coordset(**coordinates)
    readonly = result.meta.readonly
    result.meta.readonly = False
    result.meta.data_domain = "frequency"
    result.meta.isfreq = [True, True]
    result.meta.readonly = readonly
    return result
