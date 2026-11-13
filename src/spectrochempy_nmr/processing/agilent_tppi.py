"""FFT for native Agilent TPPI with explicit NMRPipe axis conventions."""

import numpy as np

import spectrochempy as scp

from .qf_interop import _fft_stage


def process_agilent_tppi(dataset, reference):
    """Return RR, RI, IR and II in F1, F2 order."""
    samples = np.asarray(dataset.data)
    if (
        tuple(dataset.dims) != ("y", "x")
        or min(samples.shape) < 2
        or not np.iscomplexobj(samples)
        or np.any(dataset.mask)
        or not np.isfinite(samples).all()
    ):
        raise ValueError("Agilent TPPI requires a complete finite complex FID")
    widths = np.asarray(dataset.meta.acquisition["spectral_widths_hz"], dtype=float)
    if widths.shape != (2,) or not np.isfinite(widths).all() or np.any(widths <= 0):
        raise ValueError("Agilent TPPI requires two positive acquisition widths")
    n1, n2 = samples.shape
    if (
        dataset.meta.acquisition.get("physical_indirect_increments") != n1
        or dataset.x.units is None
        or not dataset.x.units.is_compatible_with(scp.ur.s)
        or not np.allclose(
            dataset.x.to("s").data, np.arange(n2) / widths[1], rtol=1e-8, atol=1e-12
        )
        or not np.array_equal(dataset.y.data, np.arange(n1))
    ):
        raise ValueError("Agilent TPPI requires the original acquisition grids")

    # var2pipe -noaswap conjugates the native direct samples.
    f2 = _fft_stage(samples.conj(), -1, direct=True)
    # Zero-fill the real F1 sequences to 2N and retain bins [0, N).
    rr_ir = np.fft.rfft(f2.real, n=2 * n1, axis=0)[:n1].conj()
    ri_ii = np.fft.rfft(f2.imag, n=2 * n1, axis=0)[:n1].conj()
    planes = np.stack((rr_ir.real, ri_ii.real, rr_ir.imag, ri_ii.imag))
    labels = ("rr", "ri", "ir", "ii")

    # The half-band uses the NMRPipe TP -> ZF -> FT -real carrier grid.
    f1_hz = ((n1 - 1) / 2 - np.arange(n1)) * (widths[0] / (2 * n1))
    f2_hz = (n2 // 2 - np.arange(n2)) * (widths[1] / n2)
    values = (f1_hz, f2_hz)
    units = "Hz"
    if reference is not None:
        calibration = np.asarray(reference, dtype=float)
        if (
            calibration.shape != (2, 2)
            or not np.isfinite(calibration).all()
            or np.any(calibration[:, 0] <= 0)
        ):
            raise ValueError("TPPI reference requires (MHz, carrier ppm) for F1 and F2")
        values = tuple(
            carrier + hz / mhz
            for hz, (mhz, carrier) in zip(values, calibration, strict=True)
        )
        units = "ppm"

    result = scp.NDDataset(planes, units=dataset.units, dims=["c", "y", "x"])
    result.set_coordset(
        c=scp.Coord(labels=list(labels)),
        y=scp.Coord(values[0], units=units, title="F1", rounding=False, sigdigits=15),
        x=scp.Coord(values[1], units=units, title="F2", rounding=False, sigdigits=15),
    )
    result.meta.update(dict(dataset.meta.copy()))
    result.meta.data_domain = "frequency"
    result.meta.stage = "spectrum"
    result.meta.component_labels = labels
    result.meta.isfreq = [False, True, True]
    result.meta.iscomplex = [False, True, True]
    result.meta.td = list(planes.shape)
    result.meta.axis_roles = ("channels", "F1", "F2")
    return result
