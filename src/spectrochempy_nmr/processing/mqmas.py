"""Hybrid-domain TopSpin XFSHEAR for Bruker States acquisitions."""

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


def prepare_xfshear(
    dataset, *, size, direct_points, correct_digital_filter, echo_window=None
):
    if not isinstance(size, (tuple, list)) or len(size) != 2 or not _even_size(size[0]):
        raise ValueError("XFSHEAR size must contain two even FFT sizes (F1, F2)")
    return xf2(
        dataset,
        size=size[1],
        direct_points=direct_points,
        correct_digital_filter=correct_digital_filter,
        echo_window=echo_window,
        f1_size=size[0],
    )


def _even_size(size):
    return isinstance(size, Integral) and not isinstance(size, bool) and size % 2 == 0


def xf2(
    dataset, *, size, direct_points, correct_digital_filter, echo_window, f1_size=None
):
    """Compute complex XF2 branches for inspecting the phase before shearing."""
    from spectrochempy.core.units import ur  # noqa: PLC0415
    from spectrochempy_nmr.processing.bruker_filter import resolve_bruker_delay  # noqa: PLC0415

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
            "Read the SER without import corrections."
        )
    data = np.asarray(dataset.data)
    if not np.iscomplexobj(data) or np.any(dataset.mask) or not np.isfinite(data).all():
        raise ValueError("XFSHEAR requires finite unmasked complex samples")
    if not _even_size(size):
        raise ValueError("XF2 size must be an even FFT size")
    sx = int(size)
    sy = None if f1_size is None else int(f1_size)
    _, ny, nx = data.shape
    acquired_y, acquired_x = meta.tdeff[-2:]
    if direct_points is None:
        direct_points = min(sx, nx, acquired_x)
    if (
        not isinstance(direct_points, Integral)
        or isinstance(direct_points, bool)
        or not 2 <= direct_points <= min(sx, nx, acquired_x)
        or ny != acquired_y
        or (sy is not None and (sy < 4 or sy < ny))
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

    if echo_window is not None and (
        not isinstance(echo_window, (tuple, list))
        or len(echo_window) != 2
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            for value in echo_window
        )
        or echo_window[1] <= 0
    ):
        raise ValueError(
            "echo_window requires a finite slope and positive sigma in seconds"
        )
    if correct_digital_filter or echo_window is not None:
        acqus = meta.get("source_metadata", {}).get("acqus", {})
        recorded_delay = acquisition["digital_filter"].get("group_delay_points")
        no_decimation = (
            acqus.get("DECIM") == 1
            and acqus.get("GRPDLY") in (None, -1, 0)
            and recorded_delay in (None, 0)
        )
        delay, source = resolve_bruker_delay(dataset, 0.0 if no_decimation else None)
        if not np.isfinite(delay) or delay < 0:
            raise ValueError("XFSHEAR requires a finite nonnegative Bruker group delay")
    retained = data[..., :direct_points]
    window_metadata = None
    if echo_window is not None:
        slope, sigma = map(float, echo_window)
        t1 = np.arange(ny) / sw1
        t2 = (np.arange(direct_points) - delay) / sw2
        relative_time = t2[None, :] - slope * t1[:, None]
        retained = retained * np.exp(-0.5 * (relative_time / sigma) ** 2)
        window_metadata = {
            "slope": slope,
            "sigma_seconds": sigma,
            "group_delay_points": delay,
        }
    prepared = dataset[..., :direct_points].copy()
    prepared.data = retained.copy()
    result = prepared.fft(dim="x", size=sx, convention="topspin-xf2")
    filter_correction = {"applied": False}
    if correct_digital_filter and delay > 0:
        result.data = result.data * np.exp(-2j * np.pi * delay * np.arange(sx) / sx)
        filter_correction = {
            "applied": True,
            "group_delay_points": float(delay),
            "delay_source": source,
            "method": "frequency-domain phase ramp",
        }
    readonly = result.meta.readonly
    result.meta.readonly = False
    result.history = "Applied complex XF2 for MQMAS phase inspection"
    result.meta.update(
        {
            "stage": "mqmas_xf2",
            "data_domain": "mixed",
            "isfreq": [False, False, True],
            # The XF2 phase reference starts at zero. Initialize pk's
            # relative phase state independently of imported vendor values.
            "phased": [False, False, True],
            "phc0": [0 * ur.degree] * 3,
            "phc1": [0 * ur.degree] * 3,
            "pivot": [0] * 3,
            "exptc": [0] * 3,
            "td": list(result.shape),
            "si": list(result.shape),
            "mqmas_xf2": {
                "size": (sy, sx),
                "direct_points": int(direct_points),
                "widths_hz": (sw1, sw2),
                "sf_mhz": sf,
                "carrier_mhz": carrier,
                "digital_filter": filter_correction,
                "echo_window": window_metadata,
                "history_length": len(result.history),
            },
        }
    )
    result.meta.readonly = readonly
    return result


def _validated_xf2(dataset, *, check_phase=True):
    from spectrochempy.core.units import ur  # noqa: PLC0415

    plan = dataset.meta.get("mqmas_xf2")
    if dataset.meta.get("stage") != "mqmas_xf2" or not plan:
        raise ValueError(
            "XFSHEAR requires the full complex dataset from xf2() or prepare_xfshear()"
        )
    _, sx = plan["size"]
    sw1, sw2 = plan["widths_hz"]
    sf, carrier = plan["sf_mhz"], plan["carrier_mhz"]
    data = np.asarray(dataset.data)
    ny = dataset.shape[-2]
    if (
        tuple(dataset.dims) != ("c", "y", "x")
        or data.shape != (2, ny, sx)
        or dataset.c is None
        or dataset.c.labels is None
        or tuple(dataset.c.labels) != ("E1", "E2")
        or dataset.y is None
        or dataset.y.units != ur.s
        or dataset.x is None
        or dataset.x.units != ur.ppm
        or ny != dataset.meta.tdeff[-2]
        or not np.iscomplexobj(data)
        or np.any(dataset.mask)
        or not np.isfinite(data).all()
        or not np.allclose(
            dataset.x.data,
            (carrier - sf) * 1e6 / sf - (np.arange(sx) - sx // 2) * sw2 / (sx * sf),
            rtol=1e-10,
            atol=1e-10,
        )
        or not np.allclose(dataset.y.data, np.arange(ny) / sw1, rtol=1e-7, atol=0)
    ):
        raise ValueError(
            "XFSHEAR requires the full unmasked XF2 grid in acquired order"
        )
    if not check_phase:
        return plan
    history_length = plan.get("history_length")
    if history_length is not None and len(dataset.history) != history_length:
        raise ValueError("The XF2 processing history changed. Start from xf2().")
    phase = plan.get("f2_phase", {"phc0": 0.0, "phc1": 0.0, "pivot": 0.0})
    for name, expected in (*phase.items(), ("exptc", 0.0)):
        current = dataset.meta[name][-1]
        if getattr(current, "magnitude", current) != expected:
            raise ValueError("F2 phase was changed outside xf2p(). Start from xf2().")
    return plan


def xf2p(dataset, *, phc0, phc1, pivot, check_phase=True):
    """Apply one common F2 phase to both complex States branches."""
    plan = _validated_xf2(dataset, check_phase=check_phase)
    if "f2_phase" in plan:
        raise ValueError("F2 phase is already applied. Start from the XF2 result.")
    _, sx = plan["size"]
    for name, value in (("phc0", phc0), ("phc1", phc1), ("pivot", pivot)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
        ):
            raise ValueError(f"XFSHEAR requires a finite numerical {name}")
    if not 0 <= pivot < sx:
        raise ValueError("XFSHEAR phase pivot must be an F2 sample index")
    result = dataset.pk(dim="x", phc0=phc0, phc1=phc1, pivot=pivot, exptc=0.0, rel=True)
    readonly = result.meta.readonly
    result.meta.readonly = False
    result.meta.phc0[-1] = float(phc0) * result.meta.phc0[-1].units
    result.meta.phc1[-1] = float(phc1) * result.meta.phc1[-1].units
    result.meta.pivot[-1] = float(pivot) * result.x.units
    result.meta.exptc[-1] = 0
    result.history = "Applied common MQMAS phase in F2"
    result.meta.mqmas_xf2 = {
        **plan,
        "history_length": len(result.history),
        "f2_phase": {"phc0": float(phc0), "phc1": float(phc1), "pivot": float(pivot)},
    }
    result.meta.readonly = readonly
    return result


def replay_xfshear(dataset, *, spin, coherence_order, phc0, phc1, pivot, f1_shift):
    """Apply common F2 phase and complete XFSHEAR, including the F1 FFT."""
    phased = xf2p(dataset, phc0=phc0, phc1=phc1, pivot=pivot, check_phase=False)
    return xfshear(
        phased, size=None, spin=spin, coherence_order=coherence_order, f1_shift=f1_shift
    )


def xfshear(dataset, *, size, spin, coherence_order, f1_shift):
    """Complete XFSHEAR on phased XF2 data, including the F1 FFT."""
    from scipy.signal import hilbert  # noqa: PLC0415

    from spectrochempy.core.dataset.coord import Coord  # noqa: PLC0415
    from spectrochempy.core.units import ur  # noqa: PLC0415
    from spectrochempy_nmr.processing.topspin_xfb import _positive_fft  # noqa: PLC0415

    order, ratio, scale = _mqmas_geometry(spin, coherence_order)
    plan = _validated_xf2(dataset)
    if "f2_phase" not in plan:
        raise ValueError("Apply the common F2 phase with xf2p() before XFSHEAR")
    default_sy, sx = plan["size"]
    sy = default_sy if size is None else size
    ny = dataset.shape[-2]
    if not _even_size(sy) or sy < max(4, ny):
        raise ValueError("XFSHEAR F1 size must be even and fit all acquired increments")
    if (
        isinstance(f1_shift, bool)
        or not isinstance(f1_shift, Real)
        or not np.isfinite(f1_shift)
    ):
        raise ValueError("XFSHEAR requires a finite numerical f1_shift")
    sw1, sw2 = plan["widths_hz"]
    sf, carrier = plan["sf_mhz"], plan["carrier_mhz"]
    real_f2 = np.asarray(dataset.data).real
    centered_f2 = (np.arange(sx) - sx // 2) * sw2 / sx
    # AU f1shift is an isotropic ppm shift. Convert it using the scaled
    # carrier to obtain the additional phase ramp on the unscaled F1 grid.
    f1_shift_hz = f1_shift * carrier * scale
    phase = (
        2 * np.pi * (np.arange(ny) / sw1)[:, None] * (ratio * centered_f2 + f1_shift_hz)
    )
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

    # The AU re-references F1 after shifting the signal in the window.
    shifted_sf = sf + f1_shift_hz * 1e-6 / scale
    if not np.isfinite(shifted_sf) or shifted_sf <= 0:
        raise ValueError("XFSHEAR F1 shift gives an invalid reference frequency")
    center_f1 = (carrier - shifted_sf) * 1e6 / shifted_sf
    coords = {
        "c": Coord(labels=["S_R", "S_I"]),
        "x": dataset.x.copy(),
        "y": Coord(
            center_f1 - (np.arange(sy) - cy) * sw1 / (sy * scale * shifted_sf),
            units="ppm",
            title="isotropic chemical shift",
            rounding=False,
            sigdigits=15,
        ),
    }
    coords["y"].meta["reference_frequency"] = shifted_sf * ur.MHz
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
            "phased": [False, True, True],
            "phc0": [0 * ur.degree] * 3,
            "phc1": [0 * ur.degree] * 3,
            "pivot": [0] * 3,
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
                "direct_points": plan["direct_points"],
                "f2_phase": plan["f2_phase"],
                "f1_shift_ppm": f1_shift,
            },
        }
    )
    result.meta.readonly = readonly
    result.history = (
        "Replayed rectangular TopSpin XFSHEAR with F2 Hilbert reconstruction"
    )
    return result
