"""Hybrid-domain MQMAS and STMAS operations on normalized States components."""

from math import isfinite
from numbers import Integral
from numbers import Real

import numpy as np

from ..isotopes import Isotope, nuclear_spin


def _mqmas_geometry(spin, coherence_order):
    if (
        isinstance(spin, (bool, np.bool_))
        or not isinstance(spin, Real)
        or not isfinite(spin)
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
    spin = float(spin)
    order = int(coherence_order)
    ratio = order * (-17 * order**2 + 36 * spin * (spin + 1) - 10)
    ratio /= 9 * (4 * spin * (spin + 1) - 3)
    scale = order - ratio
    return order, ratio, scale


def _shear_spin(dataset, plan, spin, *, required=True):
    """Resolve a physical spin without inspecting archived vendor parameters."""
    acquisition = dataset.meta.get("acquisition", {})
    metadata_spins = []
    for key in ("nuclear_spin", "spin"):
        value = acquisition.get(key)
        if isinstance(value, (tuple, list)):
            value = value[-1] if value else None
        if value is not None:
            metadata_spins.append(value)
    nucleus = plan.get("observed_nucleus") or (acquisition.get("nuclei") or (None,))[-1]
    known = nuclear_spin(nucleus)
    values = [*metadata_spins, *(value for value in (known, spin) if value is not None)]
    for value in values:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not isfinite(value)
            or value < 0.5
            or not float(2 * value).is_integer()
        ):
            raise ValueError(
                "Shearing spin must be a finite positive integer or half-integer"
            )
    if values and any(value != values[0] for value in values[1:]):
        raise ValueError(
            "Shearing spin does not match the normalized spin metadata or isotope"
        )
    if not values:
        if required:
            raise ValueError(
                "No known observed isotope or spin metadata; pass spin= explicitly"
            )
        return None
    return float(values[0])


def _shear_kind(experiment):
    if experiment is None:
        kind = "mqmas"
    elif isinstance(experiment, str):
        kind = experiment.lower().replace("-", "").replace("_", "")
    else:
        kind = ""
    if kind not in ("mqmas", "3qmas", "5qmas", "stmas", "zqtrosy"):
        raise ValueError("experiment must be MQMAS, 3QMAS, 5QMAS, STMAS or ZQ-TROSY")
    return kind


def _automatic_shear(dataset, plan, experiment, spin, coherence_order):
    """Select coefficients only; quadrature reconstruction is independent."""
    if coherence_order is not None and (
        isinstance(coherence_order, (bool, np.bool_))
        or not isinstance(coherence_order, Integral)
        or coherence_order < 1
        or coherence_order % 2 == 0
    ):
        raise ValueError("coherence_order must be a positive odd integer")
    kind = _shear_kind(experiment)
    order = {"3qmas": 3, "5qmas": 5, "stmas": 1, "zqtrosy": 1}.get(kind)
    if order is not None and coherence_order is not None and coherence_order != order:
        raise ValueError(
            "coherence_order conflicts with the selected shearing experiment"
        )
    if order is None:
        order = 3 if coherence_order is None else coherence_order
    spin = _shear_spin(dataset, plan, spin, required=kind != "zqtrosy")
    if kind == "zqtrosy":
        ratio, scale = -1.0, 1.0
    elif kind == "stmas":
        if spin < 1.5 or int(2 * spin) % 2 == 0:
            raise ValueError("STMAS requires a half-integer spin of at least 3/2")
        denominator = 3 * (4 * spin * (spin + 1) - 3)
        scale = 68 / denominator
        ratio = (denominator - 68) / denominator
        nuclei = dataset.meta.get("acquisition", {}).get("nuclei") or ()
        if (
            len(nuclei) == 2
            and all(value is not None for value in nuclei)
            and nuclei[0] != nuclei[1]
        ):
            if Isotope(nuclei[0]) != Isotope(nuclei[1]):
                raise ValueError(
                    "STMAS shear requires the same observed isotope in F1 and F2"
                )
    else:
        order, ratio, scale = _mqmas_geometry(spin, order)
        kind = f"{order}qmas"
    selection = dataset.meta.get("acquisition", {}).get("multiplex_spam")
    if selection is not None and (
        order != selection["coherence_order"]
        or spin != (selection["phase_count"] - 1) / 4
    ):
        raise ValueError(
            "XFSHEAR spin and coherence order must match the Multiplex SPAM reconstruction"
        )
    return kind, spin, order, ratio, scale


def _mqecho_weights(t1, t2, slope, lb_hz):
    """Evaluate the symmetric exponential window on the acquired time grid."""
    time = t2[None, :]
    with np.errstate(over="ignore", under="ignore"):
        center = slope * t1[:, None]
        return 0.5 * (
            np.exp(-np.pi * (lb_hz * np.abs(time - center)))
            + np.exp(-np.pi * (lb_hz * np.abs(time + center)))
        )


def prepare_xfshear(
    dataset,
    *,
    size,
    direct_points,
    correct_digital_filter,
    echo_window=None,
    mqecho=None,
):
    if not isinstance(size, (tuple, list)) or len(size) != 2 or not _even_size(size[0]):
        raise ValueError("XFSHEAR size must contain two even FFT sizes (F1, F2)")
    return xf2(
        dataset,
        size=size[1],
        direct_points=direct_points,
        correct_digital_filter=correct_digital_filter,
        echo_window=echo_window,
        mqecho=mqecho,
        f1_size=size[0],
    )


def _even_size(size):
    return isinstance(size, Integral) and not isinstance(size, bool) and size % 2 == 0


def xf2(
    dataset,
    *,
    size,
    direct_points,
    correct_digital_filter,
    echo_window,
    f1_size=None,
    mqecho=None,
):
    """Compute complex XF2 branches for inspecting the phase before shearing."""
    from spectrochempy.core.units import ur  # noqa: PLC0415
    from spectrochempy.core.dataset.coord import Coord
    from .digital_filter import resolve_group_delay  # noqa: PLC0415
    from ..nmr_metadata import extract_nmr_metadata  # noqa: PLC0415
    from .fft_postprocess import _topspin_xf2_reference  # noqa: PLC0415

    meta = dataset.meta
    acquisition = meta.get("acquisition") or {}
    physical = extract_nmr_metadata(meta, ndim=dataset.ndim)
    echo_antiecho = physical.encoding == ("ECHO-ANTIECHO", "DQD")
    if (
        tuple(dataset.dims) != ("c", "y", "x")
        or meta.get("stage") != "acquisition"
        or meta.get("data_domain") != "time"
        or not physical.normalized_interop
        or physical.encoding
        not in (("STATES", "DQD"), ("STATES-TPPI", "DQD"), ("ECHO-ANTIECHO", "DQD"))
        or physical.domains != ("time", "time")
        or dataset.c is None
        or dataset.c.labels is None
        or tuple(dataset.c.labels)
        != (("first", "second") if echo_antiecho else ("E1", "E2"))
        or any(
            acquisition.get(key, {}).get("applied") is not False
            for key in ("dc_offset", "digital_filter")
        )
    ):
        raise ValueError(
            "XFSHEAR requires an uncorrected, normalized States acquisition. "
            "Read the acquisition without import corrections."
        )
    data = np.asarray(dataset.data)
    if not np.iscomplexobj(data) or np.any(dataset.mask) or not np.isfinite(data).all():
        raise ValueError("XFSHEAR requires finite unmasked complex samples")
    if not _even_size(size):
        raise ValueError("XF2 size must be an even FFT size")
    sx = int(size)
    sy = None if f1_size is None else int(f1_size)
    _, ny, nx = data.shape
    sampling = acquisition.get("sampling")
    if sampling is not None and (
        not isinstance(sampling, (tuple, list))
        or tuple(sampling) != ("regular", "regular")
    ):
        raise ValueError("XF2 requires uniform acquisition sampling")
    acquired_shape = acquisition.get("acquired_shape")
    if (
        not isinstance(acquired_shape, (tuple, list))
        or len(acquired_shape) != 2
        or any(
            isinstance(n, bool) or not isinstance(n, Integral) or n < 2
            for n in acquired_shape
        )
    ):
        raise ValueError("XF2 requires the acquired F1/F2 point counts")
    acquired_y, acquired_x = acquired_shape
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
    widths = physical.spectral_width_hz
    if widths is None or len(widths) != 2:
        raise ValueError("XF2 requires the acquisition spectral widths")
    sw1, sw2 = widths
    for axis, n, width in ((dataset.y, ny, sw1), (dataset.x, nx, sw2)):
        if (
            axis is None
            or axis.units is None
            or not axis.units.is_compatible_with(ur.s)
            or isinstance(width, (bool, np.bool_))
            or not isinstance(width, Real)
            or not np.isfinite(width)
            or width <= 0
            or not np.allclose(
                axis.data,
                (np.arange(n) / width * ur.s).to(axis.units).magnitude,
                rtol=1e-7,
                atol=0,
            )
        ):
            raise ValueError("XFSHEAR requires the original uniform acquisition grids")
    _, sf, carrier = _topspin_xf2_reference(dataset)

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
    if mqecho is not None:
        if echo_window is not None:
            raise ValueError("Specify either echo_window or mqecho")
        message = "mqecho requires a finite slope and positive LB in Hz"
        if (
            not isinstance(mqecho, (tuple, list))
            or len(mqecho) != 2
            or any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                for value in mqecho
            )
        ):
            raise ValueError(message)
        try:
            slope, lb_hz = map(float, mqecho)
        except OverflowError as exc:
            raise ValueError(message) from exc
        if not np.isfinite(slope) or not np.isfinite(lb_hz) or lb_hz <= 0:
            raise ValueError(message)
    if correct_digital_filter or echo_window is not None or mqecho is not None:
        delay, source = resolve_group_delay(dataset, None)
        if (
            isinstance(delay, (bool, np.bool_))
            or not isinstance(delay, Real)
            or not np.isfinite(delay)
            or delay < 0
        ):
            raise ValueError("XFSHEAR requires a finite nonnegative group delay")
    tppi = physical.encoding[0] == "STATES-TPPI"
    retained = data[..., :direct_points]
    if echo_antiecho:
        from .channel_interop import _rance_kay

        retained = _rance_kay(retained)
    if tppi:
        retained = retained * (1 - 2 * (np.arange(ny) % 2))[None, :, None]
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
    if mqecho is not None:
        t1 = np.arange(ny) / sw1
        t2 = (np.arange(direct_points) - delay) / sw2
        retained = retained * _mqecho_weights(t1, t2, slope, lb_hz)
        window_metadata = {
            "kind": "mqecho",
            "slope": slope,
            "lb_hz": lb_hz,
            "group_delay_points": delay,
        }
    if direct_points == 2:
        from .fft_postprocess import _topspin_xf2_axis  # noqa: PLC0415
        from .topspin_xfb import _positive_fft  # noqa: PLC0415

        result = dataset.copy()
        result.delete_coordset()
        result.data = _positive_fft(retained, sx, -1)
        result.set_coordset(
            c=dataset.c.copy(), y=dataset.y.copy(), x=_topspin_xf2_axis(dataset, sx)
        )
    else:
        prepared = dataset[..., :direct_points]
        prepared.data = retained.copy()
        prepared.meta.readonly = False
        encoding = list(
            prepared.meta.get("encoding")
            or ("CHANNELS", "UNVALIDATED:STATES", "DIRECT:DQD")
        )
        encoding[-1] = "DIRECT:DQD"
        prepared.meta.encoding = encoding
        prepared.meta.quadrature_mode = "states"
        prepared.meta.component_labels = ("E1", "E2")
        if echo_antiecho:
            prepared.c = Coord(labels=["E1", "E2"])
        result = prepared.fft(dim="x", size=sx, convention="topspin-xf2", inplace=True)
        if result.shape[-1] != sx or result.x.units != ur.ppm:
            raise ValueError("XF2 did not produce the requested F2 frequency grid")
    if result.y.units != ur.s:
        result.y = result.y.to("s")
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
    if echo_antiecho:
        result.c = Coord(labels=["E1", "E2"])
        result.meta.component_labels = ("E1", "E2")
        result.meta.echo_antiecho_reconstruction = "rance-kay"
    result.history = "Applied complex XF2 for MQMAS phase inspection"
    result.meta.update(
        {
            "stage": "mqmas_xf2",
            "quadrature_mode": "states",
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
                "acquired_shape": tuple(acquired_shape),
                "observed_nucleus": (acquisition.get("nuclei") or (None,))[-1],
                "indirect_quadrature": physical.encoding[0],
                "widths_hz": (sw1, sw2),
                "sf_mhz": sf,
                "carrier_mhz": carrier,
                "digital_filter": filter_correction,
                "echo_window": window_metadata,
                "history_length": len(result.history),
            },
        }
    )
    if tppi:
        result.meta.tppi_alternation_state = "applied"
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
    acquired = plan.get("acquired_shape") or dataset.meta.get("acquisition", {}).get(
        "acquired_shape"
    )
    if (
        not isinstance(acquired, (tuple, list))
        or len(acquired) != 2
        or any(
            isinstance(n, bool) or not isinstance(n, Integral) or n < 2
            for n in acquired
        )
        or tuple(dataset.dims) != ("c", "y", "x")
        or data.shape != (2, ny, sx)
        or dataset.c is None
        or dataset.c.labels is None
        or tuple(dataset.c.labels) != ("E1", "E2")
        or dataset.y is None
        or dataset.y.units != ur.s
        or dataset.x is None
        or dataset.x.units != ur.ppm
        or ny != acquired[0]
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
            or not np.isfinite(float(value))
        ):
            raise ValueError(f"XFSHEAR requires a finite numerical {name}")
    phc0, phc1, pivot = map(float, (phc0, phc1, pivot))
    if not 0 <= pivot < sx:
        raise ValueError("XFSHEAR phase pivot must be an F2 sample index")
    if max(abs(phc0), abs(phc1)) > np.finfo(float).max / np.pi:
        radians = np.deg2rad(phc0) + np.deg2rad(phc1) * ((np.arange(sx) - pivot) / sx)
        result = dataset.copy()
        result.data = dataset.data * np.exp(1j * radians)
    else:
        result = dataset.pk(
            dim="x", phc0=phc0, phc1=phc1, pivot=pivot, exptc=0.0, rel=True
        )
    readonly = result.meta.readonly
    result.meta.readonly = False
    result.meta.phc0[-1] = phc0 * result.meta.phc0[-1].units
    result.meta.phc1[-1] = phc1 * result.meta.phc1[-1].units
    result.meta.pivot[-1] = pivot * result.x.units
    result.meta.exptc[-1] = 0
    result.history = "Applied common MQMAS phase in F2"
    result.meta.mqmas_xf2 = {
        **plan,
        "history_length": len(result.history),
        "f2_phase": {"phc0": phc0, "phc1": phc1, "pivot": pivot},
    }
    result.meta.readonly = readonly
    return result


def replay_xfshear(dataset, *, spin, coherence_order, phc0, phc1, pivot, f1_shift):
    """Apply common F2 phase and complete XFSHEAR, including the F1 FFT."""
    phased = xf2p(dataset, phc0=phc0, phc1=phc1, pivot=pivot, check_phase=False)
    return xfshear(
        phased, size=None, spin=spin, coherence_order=coherence_order, f1_shift=f1_shift
    )


def xfshear(
    dataset,
    *,
    size,
    spin,
    coherence_order,
    f1_shift,
    ratio=None,
    f1_scale=None,
    experiment=None,
):
    """Complete XFSHEAR on phased XF2 data, including the F1 FFT."""
    from scipy.signal import hilbert  # noqa: PLC0415

    from spectrochempy.core.dataset.coord import Coord  # noqa: PLC0415
    from spectrochempy.core.units import ur  # noqa: PLC0415
    from spectrochempy_nmr.processing.topspin_xfb import _positive_fft  # noqa: PLC0415

    plan = _validated_xf2(dataset)
    acquisition = dataset.meta.get("acquisition", {})
    manual = ratio is not None or f1_scale is not None
    if manual:
        if spin is not None or coherence_order not in (None, 3):
            raise ValueError(
                "Specify physical shear coefficients or automatic MQMAS spin/order"
            )
        if (
            any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                for value in (ratio, f1_scale)
            )
            or f1_scale <= 0
        ):
            raise ValueError(
                "XFSHEAR requires a finite ratio and positive f1_scale together"
            )
        ratio, scale = float(ratio), float(f1_scale)
        order = None
        kind = _shear_kind(experiment) if experiment is not None else "explicit"
    else:
        kind, spin, order, ratio, scale = _automatic_shear(
            dataset, plan, experiment, spin, coherence_order
        )
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

    # The AU re-references quadrupolar F1, but preserves the original
    # heteronuclear F1 reference for its ZQ-TROSY branch (scale == 1).
    if kind == "zqtrosy":
        try:
            f1_carrier = acquisition["acquisition_frequency_mhz"][0]
            shifted_sf = acquisition["reference_frequency_mhz"][0]
            valid = all(
                isinstance(value, Real)
                and not isinstance(value, (bool, np.bool_))
                and np.isfinite(value)
                and value > 0
                for value in (f1_carrier, shifted_sf)
            )
        except (KeyError, TypeError, IndexError):
            valid = False
        if not valid:
            raise ValueError(
                "ZQ-TROSY requires normalized F1 acquisition and reference frequencies"
            )
    else:
        shifted_sf = sf + f1_shift_hz * 1e-6 / scale
        f1_carrier = carrier
    if not np.isfinite(shifted_sf) or shifted_sf <= 0:
        raise ValueError("XFSHEAR F1 shift gives an invalid reference frequency")
    center_f1 = (f1_carrier - shifted_sf) * 1e6 / shifted_sf
    coords = {
        "c": Coord(labels=["S_R", "S_I"]),
        "x": dataset.x.copy(),
        "y": Coord(
            center_f1 - (np.arange(sy) - cy) * sw1 / (sy * scale * shifted_sf),
            units="ppm",
            title="F1 chemical shift"
            if kind == "zqtrosy"
            else "isotropic chemical shift",
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
                "experiment": kind,
                "coefficient_source": "explicit" if manual else "automatic",
                "spin": float(spin) if spin is not None else None,
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


def stmas_shear(dataset, *, size, f1_shift, spin=None):
    """Use the shared reconstruction with automatic inner-satellite STMAS geometry."""
    if (
        isinstance(f1_shift, (bool, np.bool_))
        or not isinstance(f1_shift, Real)
        or not np.isfinite(f1_shift)
    ):
        raise ValueError("STMAS requires a finite numerical f1_shift")
    result = xfshear(
        dataset,
        size=size,
        spin=spin,
        coherence_order=None,
        f1_shift=-f1_shift,
        experiment="stmas",
    )
    readonly = result.meta.readonly
    result.meta.readonly = False
    shear = result.meta.mqmas_shearing
    result.meta.stmas_shear = {
        "ratio": shear["ratio"],
        "scale_divisor": shear["scale_divisor"],
        "f1_shift_ppm": float(f1_shift),
        "f2_phase": shear["f2_phase"],
        "digital_filter": result.meta.mqmas_xf2["digital_filter"],
    }
    result.meta.readonly = readonly
    result.history = "Applied STMAS through the common XFSHEAR reconstruction"
    return result
