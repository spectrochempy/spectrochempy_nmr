"""Coordinate-aware shearing of frequency-domain MQMAS maps."""

from numbers import Integral
from numbers import Real

import numpy as np


def _mqmas_geometry(spin, coherence_order):
    if (
        isinstance(spin, (bool, np.bool_))
        or not isinstance(spin, Real)
        or not np.isfinite(spin)
        or spin < 1.5
        or not float(2 * spin).is_integer()
        or int(2 * spin) % 2 == 0
    ):
        raise ValueError("MQMAS requires a half-integer spin of at least 3/2")
    if (
        isinstance(coherence_order, (bool, np.bool_))
        or not isinstance(coherence_order, Integral)
        or coherence_order < 3
        or coherence_order % 2 == 0
        or coherence_order > 2 * spin
    ):
        raise ValueError("MQMAS requires an odd coherence order from 3 to 2*spin")
    order = int(coherence_order)
    ratio = order * (-17 * order**2 + 36 * spin * (spin + 1) - 10)
    ratio /= 9 * (4 * spin * (spin + 1) - 3)
    scale = order - ratio
    return order, ratio, scale


def shear_mqmas(dataset, *, spin, coherence_order=3, pivot=0.0):
    """Shear an MQMAS spectrum and rescale its indirect frequency axis.

    The transformation is f_iso = (f_MQ - R * (f_SQ - pivot)) / (p - R).
    Both axes must use Hz or ppm with an explicit frequency reference.
    A numerical pivot is in Hz. Fourier interpolation is periodic along F1.
    The input must already be reconstructed and Fourier transformed.
    Phasing and digital-filter correction precede this operation.

    For spin 5/2 and 3Q, R is 19/12 and the scale divisor is 17/12.
    See Fernandez and Pruski (2011), DOI 10.1007/128_2011_141.
    """
    import spectrochempy as scp  # noqa: PLC0415

    order, ratio, scale = _mqmas_geometry(spin, coherence_order)
    if dataset.meta.get("mqmas_shearing"):
        raise ValueError("MQMAS shearing has already been applied")
    if set(dataset.dims) not in ({"y", "x"}, {"c", "y", "x"}):
        raise ValueError("MQMAS requires x/y spectral axes and optional c planes")
    if dataset.coordset is None:
        raise ValueError("MQMAS requires two calibrated spectral grids")
    if "c" in dataset.dims:
        labels = dataset.c.labels if dataset.c is not None else None
        layout = tuple(labels) if labels is not None else ()
        real_planes = layout == ("rr", "ri", "ir", "ii") and not np.iscomplexobj(
            dataset.data
        )
        complex_planes = layout == ("S_R", "S_I") and np.iscomplexobj(dataset.data)
        if dataset.meta.get("component_axis") != "c" or not (
            real_planes or complex_planes
        ):
            raise ValueError("MQMAS requires spectral rr/ri/ir/ii or S_R/S_I planes")
    if np.any(dataset.mask) or not np.isfinite(dataset.data).all():
        raise ValueError("MQMAS requires finite unmasked spectra")

    coordinates = {}
    for dim in ("y", "x"):
        axis = dataset.coordset[dim]
        if axis is None or axis.size < 2:
            raise ValueError("MQMAS requires two calibrated spectral grids")
        if axis.units not in (scp.ur.Hz, scp.ur.ppm):
            raise ValueError("MQMAS axes must be in Hz or referenced ppm")
        try:
            hz = axis.to("Hz")
        except (ValueError, TypeError) as exc:
            raise ValueError("MQMAS ppm axes require a frequency reference") from exc
        values = np.asarray(hz.data)
        steps = np.diff(values)
        if (
            not np.isfinite(values).all()
            or steps[0] == 0
            or not np.allclose(steps, steps[0], rtol=1e-7, atol=1e-10)
        ):
            raise ValueError("MQMAS requires uniform monotonic spectral grids")
        coordinates[dim] = hz
    try:
        pivot_hz = (
            float(pivot.to("Hz").magnitude) if hasattr(pivot, "to") else float(pivot)
        )
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("MQMAS pivot must be a finite frequency") from exc
    if isinstance(pivot, (bool, np.bool_)) or not np.isfinite(pivot_hz):
        raise ValueError("MQMAS pivot must be a finite frequency")

    data = np.asarray(dataset.data)
    y_axis = dataset.dims.index("y")
    x_axis = dataset.dims.index("x")
    work = np.moveaxis(data, (y_axis, x_axis), (-2, -1))
    y = coordinates["y"].data
    shifts = ratio * (coordinates["x"].data - pivot_hz) / (y[1] - y[0])
    ramp = np.exp(2j * np.pi * np.fft.fftfreq(len(y))[:, None] * shifts)
    if "c" in dataset.dims and np.iscomplexobj(data) and len(y) % 2 == 0:
        # The even-length Nyquist bin represents both signs for each real
        # plane. Use its cosine response in the packed representation too.
        ramp[len(y) // 2] = ramp[len(y) // 2].real
    shifted = np.fft.ifft(np.fft.fft(work, axis=-2) * ramp, axis=-2)
    if not np.iscomplexobj(data):
        shifted = shifted.real
    if not np.isfinite(shifted).all():
        raise ValueError("MQMAS shearing produced nonfinite samples")

    result = dataset.copy()
    result.data = np.moveaxis(shifted, (-2, -1), (y_axis, x_axis))
    isotropic = scp.Coord(
        y / scale,
        units="Hz",
        title="isotropic frequency",
        rounding=False,
        sigdigits=15,
    )
    isotropic.meta.update(dict(coordinates["x"].meta.copy()))
    if dataset.coordset["y"].units == scp.ur.ppm:
        isotropic.ito("ppm")
        isotropic.title = "isotropic chemical shift"
    result.coordset["y"] = isotropic
    readonly = result.meta.readonly
    result.meta.readonly = False
    result.meta.mqmas_shearing = {
        "spin": float(spin),
        "coherence_order": order,
        "ratio": float(ratio),
        "scale_divisor": float(scale),
        "pivot_hz": pivot_hz,
        "boundary": "periodic",
    }
    result.meta.readonly = readonly
    result.history = "Applied MQMAS shearing and isotropic F1 scaling"
    return result


def replay_xfshear(dataset, *, spin, coherence_order, size, direct_points):
    """Process an uncorrected Bruker States SER with the rectangular AU recipe."""
    from scipy.signal import hilbert  # noqa: PLC0415

    from spectrochempy.core.dataset.coord import Coord  # noqa: PLC0415
    from spectrochempy.core.units import ur  # noqa: PLC0415
    from spectrochempy_nmr.processing.topspin_xfb import _positive_fft  # noqa: PLC0415

    order, ratio, scale = _mqmas_geometry(spin, coherence_order)
    meta = dataset.meta
    acquisition = meta.get("acquisition") or {}
    if (
        tuple(dataset.dims) != ("c", "y", "x")
        or meta.get("stage") != "acquisition"
        or meta.get("data_domain") != "time"
        or meta.get("normalization_profile") != "bruker-ser-states-2d-v1"
        or dataset.c is None
        or dataset.c.labels is None
        or tuple(dataset.c.labels) != ("E1", "E2")
        or any(
            acquisition.get(key, {}).get("applied") is not False
            for key in ("dc_offset", "digital_filter")
        )
    ):
        raise ValueError(
            "XFSHEAR requires an uncorrected, normalized Bruker States SER. "
            "Read it with remove_dc_offset=False and remove_digital_filter=False."
        )
    data = np.asarray(dataset.data)
    if not np.iscomplexobj(data) or np.any(dataset.mask) or not np.isfinite(data).all():
        raise ValueError("XFSHEAR requires finite unmasked complex samples")
    if (
        not isinstance(size, (tuple, list))
        or len(size) != 2
        or any(
            isinstance(n, bool) or not isinstance(n, Integral) or n % 2 for n in size
        )
    ):
        raise ValueError("XFSHEAR size must contain two even FFT sizes (F1, F2)")
    sy, sx = map(int, size)
    _, ny, nx = data.shape
    acquired_y, acquired_x = meta.tdeff[-2:]
    if direct_points is None:
        direct_points = min(sx, nx, acquired_x)
    if (
        not isinstance(direct_points, Integral)
        or isinstance(direct_points, bool)
        or not 2 <= direct_points <= min(sx, nx, acquired_x)
        or not (4 <= sy and ny == acquired_y <= sy)
    ):
        raise ValueError(
            "XFSHEAR requires all F1 increments and enough acquired F2 samples"
        )
    sw1, sw2 = (float(w.to("Hz").magnitude) for w in meta.sw_h[-2:])
    for axis, n, width in ((dataset.y, ny, sw1), (dataset.x, nx, sw2)):
        if (
            not np.isfinite(width)
            or width <= 0
            or not np.allclose(
                axis.data,
                (np.arange(n) / width * ur.s).to(axis.units).magnitude,
                rtol=1e-7,
                atol=0,
            )
        ):
            raise ValueError("XFSHEAR requires the original uniform acquisition grids")
    sf = float(meta.sf[-1].to("MHz").magnitude)
    carrier = float(meta.sfo1[-1].to("MHz").magnitude)
    if not np.isfinite(sf) or sf <= 0 or not np.isfinite(carrier) or carrier <= 0:
        raise ValueError("XFSHEAR requires the direct processing reference")

    real_f2 = _positive_fft(data[..., :direct_points], sx, -1).real
    centered_f2 = (np.arange(sx) - sx // 2) * sw2 / sx
    phase = 2 * np.pi * ratio * (np.arange(ny) / sw1)[:, None] * centered_f2
    rotated = (real_f2[0] + 1j * real_f2[1]) * np.exp(1j * phase)
    branches = np.stack((rotated.real, rotated.imag))

    # XHT2 uses zero extension to twice SI. With the rectangular window
    # and phase, baseline and PKNL disabled, XIF2 and the final F2 FFT cancel.
    reconstructed = hilbert(branches, N=2 * sx, axis=-1)[..., :sx].conj()
    # F1 and F2 are independent complex dimensions. Keep their quadratures
    # separate before applying the F1 first-point weight and transform.
    pairs = np.stack(
        (
            reconstructed[0].real + 1j * reconstructed[1].real,
            reconstructed[0].imag + 1j * reconstructed[1].imag,
        )
    )
    pairs[:, 0] *= 0.5
    direct_real, direct_imag = _positive_fft(pairs, sy, -2)
    rr = direct_real.real
    cy, cx = sy // 2, sx // 2
    rr[cy, cx] = (rr[cy - 1, cx] + rr[cy + 1, cx]) / 2
    spectrum = np.stack(
        (rr + 1j * direct_imag.real, -direct_real.imag - 1j * direct_imag.imag)
    )
    if not np.isfinite(spectrum).all():
        raise ValueError("XFSHEAR produced nonfinite spectral values")

    center = (carrier - sf) * 1e6 / sf
    coords = {
        "c": Coord(labels=["S_R", "S_I"]),
        "x": Coord(
            center - centered_f2 / sf,
            units="ppm",
            title="F2 chemical shift",
            rounding=False,
            sigdigits=15,
        ),
        "y": Coord(
            center - (np.arange(sy) - cy) * sw1 / (sy * scale * sf),
            units="ppm",
            title="isotropic chemical shift",
            rounding=False,
            sigdigits=15,
        ),
    }
    for dim in ("y", "x"):
        coords[dim].meta["reference_frequency"] = sf * ur.MHz
    result = dataset.copy()
    result.delete_coordset()
    result.data = spectrum
    result.set_coordset(**coords)
    readonly = result.meta.readonly
    result.meta.readonly = False
    result.meta.update(
        {
            "data_domain": "frequency",
            "stage": "spectrum",
            "isfreq": [False, True, True],
            "iscomplex": [False, True, True],
            "component_labels": ("S_R", "S_I"),
            "quadrature_mode": "complex-channel-pair-2d",
            "encoding": ["CHANNELS", "STATES", "DIRECT:DQD"],
            "si": list(result.shape),
            "td": list(result.shape),
            "mqmas_shearing": {
                "replay": "topspin-xfshear",
                "spin": float(spin),
                "coherence_order": order,
                "ratio": float(ratio),
                "scale_divisor": float(scale),
                "hilbert_fft_size": 2 * sx,
                "direct_points": int(direct_points),
            },
        }
    )
    result.meta.readonly = readonly
    result.history = (
        "Replayed rectangular TopSpin XFSHEAR with F2 Hilbert reconstruction"
    )
    return result
