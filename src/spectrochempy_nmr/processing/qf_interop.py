"""FFT-only processing for an explicitly normalized QF acquisition."""

from numbers import Integral, Real

import numpy as np


def _fft_stage(values, axis, *, direct):
    if direct:
        # NMRPipe FT -nodf uses the unnormalized positive-sign transform.
        transformed = np.fft.ifft(values, axis=axis) * values.shape[axis]
    else:
        # The qualified QF indirect step uses NMRPipe FT -neg.
        transformed = np.fft.fft(values, axis=axis).conj()
    return np.fft.fftshift(transformed, axes=axis)


def process_qf_fft(dataset, widths, *, size=None):
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

    if size is None:
        sizes = data.shape
    elif isinstance(size, Integral) and not isinstance(size, (bool, np.bool_)):
        sizes = (size, size)
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        sizes = tuple(size)
    else:
        raise ValueError("QF size must be an integer or a pair of FFT sizes")
    if any(
        isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < acquired
        for n, acquired in zip(sizes, data.shape, strict=True)
    ):
        raise ValueError("QF FFT sizes must retain all acquired points")
    acquisition = dataset.meta.get("acquisition", {})
    offsets = acquisition.get("frequency_offset_hz", (0.0, 0.0))
    frequencies = acquisition.get("acquisition_frequency_mhz")
    if (
        not isinstance(offsets, (tuple, list))
        or len(offsets) != 2
        or any(
            isinstance(v, (bool, np.bool_))
            or not isinstance(v, Real)
            or not np.isfinite(v)
            for v in offsets
        )
    ):
        raise ValueError(
            "QF frequency offsets must be two finite physical values in Hz"
        )
    edges = acquisition.get("nyquist_edge", ("positive", "positive"))
    if (
        not isinstance(edges, (tuple, list))
        or len(edges) != 2
        or any(edge not in ("positive", "negative") for edge in edges)
    ):
        raise ValueError("QF Nyquist edges must be positive or negative")
    coordinates = {}
    for dim, acquired, size, width, offset, edge in zip(
        dataset.dims, data.shape, sizes, widths, offsets, edges, strict=True
    ):
        axis = dataset.coordset[dim]
        if (
            acquired < 2
            or axis.units != scp.ur.s
            or isinstance(width, (bool, np.bool_))
            or not isinstance(width, (int, float, np.integer, np.floating))
            or not np.isfinite(width)
            or width <= 0
            or not np.allclose(
                axis.data, np.arange(acquired) / width, rtol=1e-9, atol=0
            )
        ):
            raise ValueError(
                "normalized 2D QF FFT requires the declared uniform time grids"
            )
        # Carrier-relative NMRPipe grid; the acquisition does not establish a chemical zero.
        shift = int(edge == "negative" and size % 2 == 0)
        hz = offset + (size // 2 - shift - np.arange(size)) * (width / size)
        coordinates[dim] = scp.Coord(
            hz, units="Hz", title=axis.title, rounding=False, sigdigits=15
        )

    if frequencies is not None:
        if (
            not isinstance(frequencies, (tuple, list))
            or len(frequencies) != 2
            or any(
                isinstance(f, (bool, np.bool_))
                or not isinstance(f, Real)
                or not np.isfinite(f)
                or f <= 0
                for f in frequencies
            )
        ):
            raise ValueError("QF observation frequencies must be positive MHz values")
        for dim, frequency in zip(dataset.dims, frequencies, strict=True):
            coordinates[dim].meta["acquisition_frequency"] = frequency * scp.ur.MHz
            coordinates[dim].meta["reference_frequency"] = frequency * scp.ur.MHz
    f2 = tuple(dataset.meta.axis_roles).index("F2")
    data = np.pad(
        data, tuple((0, size - n) for size, n in zip(sizes, data.shape, strict=True))
    )
    spectrum = _fft_stage(_fft_stage(data, f2, direct=True), 1 - f2, direct=False)
    for axis, (size, edge) in enumerate(zip(sizes, edges, strict=True)):
        if edge == "negative" and size % 2 == 0:
            spectrum = np.roll(spectrum, -1, axis=axis)
    result = dataset.copy()
    if tuple(sizes) != dataset.shape:
        result = scp.NDDataset(
            spectrum,
            dims=list(dataset.dims),
            units=dataset.units,
            name=dataset.name,
            title=dataset.title,
        )
        result.meta = dataset.meta.copy()
        result.filename = dataset.filename
        result.origin = dataset.origin
        result.description = dataset.description
    else:
        result.data = spectrum
    result.set_coordset(**coordinates)
    readonly = result.meta.readonly
    result.meta.readonly = False
    result.meta.data_domain = "frequency"
    result.meta.interop_domains = ("frequency", "frequency")
    result.meta.stage = "spectrum"
    result.meta.isfreq = [True, True]
    result.meta.readonly = readonly
    return result
