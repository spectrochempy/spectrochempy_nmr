"""Replay the qualified uniform Bruker XFB profiles."""

import numpy as np
from numpy.polynomial.legendre import legvander

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.units import ur
from spectrochempy_nmr.processing.channel_interop import _cross_pairs
from spectrochempy_nmr.processing.channel_interop import _rance_kay
from spectrochempy_nmr.processing.linear_prediction import _burg_predict_after

_PROFILES = {
    "bruker-ser-qf-2d-v1": "qf",
    "bruker-ser-states-tppi-2d-v1": "states-tppi",
    "bruker-ser-echo-antiecho-2d-v1": "echo-antiecho",
}


def _check_axis(params, points, words_per_point, ft_mode, phase_mode):
    if params.get("TDoff") != 0:
        raise ValueError("TopSpin XFB requires TDoff=0 on both axes")
    size = params.get("SI")
    if (
        not isinstance(size, int)
        or size < points
        or size % 2
        or params.get("FTSIZE") != size
        or params.get("STSI") != size
        or params.get("STSR") != 0
        or params.get("TDeff") != points * words_per_point
        or params.get("FT_mod") != ft_mode
        or params.get("PH_mod") != phase_mode
        or params.get("REVERSE") is not False
        or params.get("WDW") not in (0, 1, 3, 4)
    ):
        raise ValueError(
            "TopSpin XFB requires a full grid and supported window and phase"
        )
    for key in ("FCOR", "PHC0", "PHC1", "OFFSET", "SW_p", "SF"):
        value = params.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"TopSpin XFB requires a finite {key}")
    if min(params["SW_p"], params["SF"]) <= 0:
        raise ValueError("TopSpin XFB requires a calibrated processing grid")
    if params["WDW"] == 1:
        lb = params.get("LB")
        if not isinstance(lb, (int, float)) or not np.isfinite(lb):
            raise ValueError("TopSpin XFB requires finite exponential LB")
    if params["WDW"] in (3, 4):
        shift = params.get("SSB")
        if not isinstance(shift, (int, float)) or not np.isfinite(shift) or shift < 0:
            raise ValueError("TopSpin XFB requires a valid sine window shift")


def _window(params, points, width, delay=0):
    mode = params["window_mode"]
    if mode == 1:
        time = (np.arange(points) - delay) / width
        with np.errstate(over="ignore"):
            weights = np.exp(-np.pi * time * params["line_broadening_hz"])
        if not np.isfinite(weights).all():
            raise ValueError("TopSpin XFB exponential window overflowed")
    elif mode in (3, 4):
        shift = params["sine_shift"]
        phase = np.pi / shift if shift >= 2 else 0
        position = (np.arange(points) - delay) / (points - delay + 0.5)
        weights = np.sin(phase + (np.pi - phase) * position)
        if mode == 4:
            weights **= 2
    else:
        weights = np.ones(points)
    weights[0] *= params["first_point_scale"]
    return weights


def _f1_prediction(params, points, words_per_point):
    mode = params.get("ME_mod")
    if mode == 0:
        return points, 0
    if mode != 2:
        raise ValueError("TopSpin XFB supports only LPfr linear prediction on F1")
    words = params.get("LPBIN")
    coefficients = params.get("NCOEF")
    if (
        not isinstance(words, int)
        or words % words_per_point
        or words <= points * words_per_point
        or not isinstance(coefficients, int)
        or coefficients % 2
        or not 2 <= coefficients <= points
    ):
        raise ValueError("TopSpin XFB requires valid F1 LPBIN and even NCOEF for LPfr")
    return words // words_per_point, coefficients // 2


def _prepare_f1(data, window, points, order):
    if order:
        data = _burg_predict_after(data.swapaxes(-2, -1), points, order)
        data = data.swapaxes(-2, -1)
    return data * window


def _check_baseline(params, supported, points):
    mode = params.get("BC_mod")
    if mode not in supported:
        raise ValueError(
            "TopSpin XFB baseline correction mode is not supported on this axis"
        )
    if mode in (1, 2) and (points < 4 or points % 4):
        raise ValueError("TopSpin XFB tail-mean baseline requires full quarters")
    if mode in (3, 4, 5, 6):
        offset = params.get("COROFFS")
        if mode in (3, 4) and points < 6:
            raise ValueError(
                "TopSpin XFB polynomial baseline requires at least six points"
            )
        if (
            not isinstance(offset, (int, float))
            or not np.isfinite(offset)
            or (mode in (3, 5) and offset != 0)
        ):
            raise ValueError(
                "TopSpin XFB requires a valid COROFFS for the baseline mode"
            )
    if mode in (5, 6):
        width = params.get("BCFW")
        if not isinstance(width, (int, float)) or not np.isfinite(width) or width <= 0:
            raise ValueError("TopSpin XFB requires a positive finite BCFW")
        if points < 8:
            raise ValueError(
                "TopSpin XFB Gaussian baseline requires at least eight points"
            )


def _gaussian_baseline(data, radius):
    """Reproduce XFB convolution and linear extension at both trace ends."""
    points = data.shape[-1]
    step = 2 * radius
    last = points - radius - 2
    kernel = np.exp(-4 * (np.arange(-radius, radius + 1) / radius) ** 2)
    kernel /= kernel.sum()
    interior = np.apply_along_axis(np.convolve, -1, data[..., :-1], kernel, "valid")
    first_value, last_value = interior[..., :1], interior[..., -1:]
    left_anchor = interior[..., radius : radius + 1]
    right_index = points - step - radius
    right_anchor = interior[..., right_index : right_index + 1] if radius > 1 else 0
    left = (
        first_value + (first_value - left_anchor) * (radius - np.arange(radius)) / step
    )
    right = (
        last_value
        + (last_value - right_anchor) * (np.arange(last + 1, points) - last) / step
    )
    return np.concatenate((left, interior, right), axis=-1)


def _gaussian_radius(params, points):
    width_ppm = params["spectral_width_hz"] / params["reference_frequency_mhz"]
    return int(
        min(
            points // 4, max(1, 0.625 * width_ppm / params["baseline_filter_width_ppm"])
        )
    )


def _subtract_baseline(data, mode, cycles_per_point=0, radius=0):
    """Subtract the measured XFB time-domain baseline."""
    points = data.shape[-1]
    if mode in (1, 2):
        return data - data[..., 3 * points // 4 :].mean(axis=-1, keepdims=True)
    ramp = (
        np.exp(2j * np.pi * (cycles_per_point % 1) * np.arange(points))
        if cycles_per_point
        else 1
    )
    shifted = data * ramp
    if radius:
        baseline = _gaussian_baseline(shifted, radius)
    else:
        basis = legvander(np.linspace(-1, 1, points), 5)
        factors = np.array([1, 3, 5, 7, 9, 121 / 63]) / points
        baseline = ((shifted @ basis) * factors) @ basis.T
    return data - baseline * np.conj(ramp)


def _correct_f1_baseline(data, params, profile, width):
    """Use TopSpin's scalar word order for the single-channel F1 modes."""
    baseline = params["baseline_mode"]
    if not baseline:
        return data
    traces = data.swapaxes(-2, -1)
    radius = _gaussian_radius(params, traces.shape[-1]) if baseline in (5, 6) else 0
    if baseline in (1, 3, 5):
        if profile == "qf":
            traces = traces.conj()
        elif profile == "echo-antiecho":
            traces = 1j * traces
        words = np.ascontiguousarray(traces).view(traces.real.dtype)
        corrected = _subtract_baseline(words, baseline, radius=radius).view(
            traces.dtype
        )
        if profile == "qf":
            corrected = corrected.conj()
        elif profile == "echo-antiecho":
            corrected = -1j * corrected
    else:
        sign = 1 if profile == "qf" else -1
        cycles_per_point = (
            sign * params["baseline_offset_hz"] / width if baseline in (4, 6) else 0
        )
        corrected = _subtract_baseline(traces, baseline, cycles_per_point, radius)
    return corrected.swapaxes(-2, -1)


def _positive_fft(data, size, axis):
    return np.fft.fftshift(np.fft.ifft(data, n=size, axis=axis) * size, axes=axis)


def _processed_axis(params, source_axis):
    bins = np.arange(params["fft_size"]) / params["fft_size"]
    axis = Coord(
        params["first_point_ppm"]
        - bins * (params["spectral_width_hz"] / params["reference_frequency_mhz"]),
        units="ppm",
        title=source_axis.title,
        rounding=False,
        sigdigits=15,
    )
    axis.meta["reference_frequency"] = params["reference_frequency_mhz"] * ur.MHz
    return axis


def replay_xfb(source):
    meta = source.meta
    mode = _PROFILES.get(meta.get("normalization_profile"))
    qf = mode == "qf"
    dims = ("y", "x") if qf else ("c", "y", "x")
    if (
        mode is None
        or tuple(source.dims) != dims
        or meta.get("data_domain") != "time"
        or tuple(meta.get("interop_physical_dimensions", ())) != ("F1", "F2")
    ):
        raise ValueError("TopSpin XFB replay requires a normalized uniform Bruker SER")
    acquisition = meta["acquisition"]
    if any(
        acquisition.get(key, {}).get("applied")
        for key in ("dc_offset", "digital_filter")
    ):
        raise ValueError("TopSpin XFB requires an uncorrected SER")
    data = np.asarray(source.data)
    if not np.iscomplexobj(data) or np.any(source.mask) or not np.isfinite(data).all():
        raise ValueError("TopSpin XFB requires finite unmasked complex samples")
    if not qf:
        labels = ("E1", "E2") if mode == "states-tppi" else ("first", "second")
        if (
            data.shape[0] != 2
            or source.c is None
            or source.c.labels is None
            or tuple(source.c.labels) != labels
        ):
            raise ValueError("TopSpin XFB requires both branches in acquired order")
    ny, nx = data.shape[-2:]
    widths = acquisition["spectral_width_hz"]
    for axis, size, width in zip((source.y, source.x), (ny, nx), widths, strict=True):
        if axis.units != ur.s or not np.allclose(
            axis.data, np.arange(size) / width, rtol=1e-9, atol=0
        ):
            raise ValueError("TopSpin XFB requires the original calibrated time grids")

    plan = _build_xfb_plan(meta.get("source_metadata", {}), mode, (ny, nx), widths)
    return _execute_xfb(source, data, plan)


def _xfb_axis_plan(params):
    """Translate validated TopSpin fields to execution parameters."""
    window = int(params["WDW"])
    baseline = int(params["BC_mod"])
    return {
        "window_mode": window,
        "line_broadening_hz": float(params["LB"]) if window == 1 else None,
        "sine_shift": float(params["SSB"]) if window in (3, 4) else None,
        "first_point_scale": float(params["FCOR"]),
        "spectral_width_hz": float(params["SW_p"]),
        "reference_frequency_mhz": float(params["SF"]),
        "baseline_filter_width_ppm": float(params["BCFW"])
        if baseline in (5, 6)
        else None,
        "baseline_mode": baseline,
        "baseline_offset_hz": float(params["COROFFS"]) if baseline in (4, 6) else None,
        "first_point_ppm": float(params["OFFSET"]),
        "fft_size": params["SI"],
        "phase_zero_deg": float(params["PHC0"]),
        "phase_first_deg": float(params["PHC1"]),
        "digital_filter_phase": params["PKNL"],
    }


def _build_xfb_plan(status, mode, shape, widths):
    """Interpret a TopSpin recipe without applying any processing."""
    ny, nx = shape
    qf = mode == "qf"
    y, x = status.get("proc2s", {}), status.get("procs", {})
    if x.get("ME_mod") != 0:
        raise ValueError("TopSpin XFB linear prediction is only supported on F1")
    words_per_point = 1 if qf else 2
    f1_points, lp_order = _f1_prediction(y, ny, words_per_point)
    _check_axis(
        y, f1_points, words_per_point, 4 if mode == "states-tppi" else 6, 2 if qf else 1
    )
    _check_axis(x, nx, 2, 6, 0 if qf else 1)
    if y["SI"] < 4:
        raise ValueError("TopSpin XFB requires both neighbors of the F1 center bin")
    if x["FCOR"] != 1:
        raise ValueError("TopSpin XFB requires FCOR=1 on F2")
    if y.get("PKNL") is not False or not isinstance(x.get("PKNL"), bool):
        raise ValueError("TopSpin XFB supports PKNL only on F2")
    delay = status.get("acqus", {}).get("GRPDLY")
    if (
        not isinstance(delay, (int, float))
        or not np.isfinite(delay)
        or not 0 <= delay < nx - 1
    ):
        raise ValueError("TopSpin XFB requires a known direct group delay")

    origin = int(np.ceil(delay))
    if origin >= nx - 1:
        raise ValueError("TopSpin XFB group delay leaves fewer than two direct points")
    _check_baseline(y, (0, 1, 2, 3, 4, 5, 6), ny)
    _check_baseline(x, (0, 4, 6), nx - origin)
    return {
        "mode": mode,
        "widths_hz": tuple(float(width) for width in widths),
        "f1_points": f1_points,
        "f1_prediction_order": lp_order,
        "group_delay_points": delay,
        "f2_time_origin_points": origin,
        "f1": _xfb_axis_plan(y),
        "f2": _xfb_axis_plan(x),
    }


def _execute_xfb(source, data, plan):
    """Run an already validated XFB plan."""
    mode = plan["mode"]
    qf = mode == "qf"
    _, nx = data.shape[-2:]
    widths = plan["widths_hz"]
    f1_points = plan["f1_points"]
    lp_order = plan["f1_prediction_order"]
    delay = plan["group_delay_points"]
    origin = plan["f2_time_origin_points"]
    y, x = plan["f1"], plan["f2"]
    f2_input = data.copy() if x["baseline_mode"] else data
    if x["baseline_mode"]:
        radius = _gaussian_radius(x, nx - origin) if x["baseline_mode"] == 6 else 0
        f2_input[..., origin:] = _subtract_baseline(
            data[..., origin:],
            x["baseline_mode"],
            -x["baseline_offset_hz"] / widths[1],
            radius,
        )
    f2_input = f2_input * _window(x, nx, widths[1], origin)
    if mode == "echo-antiecho":
        f2_input = _rance_kay(f2_input)
    sx, sy = x["fft_size"], y["fft_size"]
    f2 = _positive_fft(f2_input, sx, -1)
    phase_x = (
        x["phase_zero_deg"]
        + (x["phase_first_deg"] + 360 * delay * x["digital_filter_phase"])
        * np.arange(sx)
        / sx
    )
    f2 *= np.exp(-1j * np.deg2rad(phase_x))
    f1_window = _window(y, f1_points, widths[0])[:, None]
    if qf:
        f1_input = _correct_f1_baseline(f2, y, mode, widths[0])
        f1_input = _prepare_f1(f1_input, f1_window, f1_points, lp_order)
        f1 = np.fft.fftshift(np.fft.fft(f1_input, n=sy, axis=-2), axes=-2)
        spectrum = np.abs(f1)
        rr = spectrum
    else:
        cross = _correct_f1_baseline(_cross_pairs(f2), y, mode, widths[0])
        if mode == "states-tppi":
            cross[:, 1::2] *= -1
        cross = _prepare_f1(cross, f1_window, f1_points, lp_order)
        a, b = _positive_fft(cross, sy, -2)
        if mode == "states-tppi":
            rr_ir, ri_ii = a.real - 1j * b.real, -a.imag + 1j * b.imag
        else:
            rr_ir, ri_ii = a.imag - 1j * b.imag, a.real - 1j * b.real
        phase_y = y["phase_zero_deg"] + y["phase_first_deg"] * np.arange(sy) / sy
        rotation = np.exp(1j * np.deg2rad(phase_y))[:, None]
        # Rotate F1 in pairs (2rr, 2ri) and (2ir, 2ii).
        real = (rr_ir.real + 1j * ri_ii.real) * rotation
        imag = (rr_ir.imag + 1j * ri_ii.imag) * rotation
        spectrum = np.stack((real.real + 1j * imag.real, real.imag + 1j * imag.imag))
        rr = spectrum[0].real

    cy, cx = sy // 2, sx // 2
    rr[cy, cx] = (rr[cy - 1, cx] + rr[cy + 1, cx]) / 2
    if not np.isfinite(spectrum).all():
        raise ValueError("TopSpin XFB replay produced nonfinite spectral values")
    result = source.copy()
    result.delete_coordset()
    result.data = spectrum
    coords = {
        "y": _processed_axis(y, source.y),
        "x": _processed_axis(x, source.x),
    }
    if not qf:
        coords["c"] = Coord(labels=["S_R", "S_I"])
    result.set_coordset(**coords)
    readonly = result.meta.readonly
    result.meta.readonly = False
    result.meta.data_domain = "frequency"
    result.meta.stage = "spectrum"
    result.meta.isfreq = [True, True] if qf else [False, True, True]
    result.meta.iscomplex = [False, False] if qf else [False, True, True]
    result.meta.si = list(result.shape)
    result.meta.td = list(result.shape)
    result.meta.fft_presentation = None
    result.meta.phased = result.meta.isfreq.copy()
    if result.meta.get("encoding"):
        result.meta.encoding = [
            encoding.removeprefix("PENDING-F2:") for encoding in result.meta.encoding
        ]
    if not qf:
        result.meta.component_labels = ("S_R", "S_I")
        result.meta.quadrature_mode = "complex-channel-pair-2d"
        if mode == "states-tppi":
            result.meta.tppi_alternation_state = "applied"
        else:
            result.meta.echo_antiecho_reconstruction = "bruker-rance-kay"
    result.meta.readonly = readonly
    applied = {
        "replay": "topspin",
        "command": "xfb",
        "size": (sy, sx),
        "window": (y["window_mode"], x["window_mode"]),
        "group_delay_points": delay,
        "f2_time_origin_points": origin,
        "pknl": x["digital_filter_phase"],
        "phase": {
            "f1": (y["phase_zero_deg"], y["phase_first_deg"]),
            "f2": (x["phase_zero_deg"], x["phase_first_deg"]),
        },
        "axis_calibration": "procs",
    }
    if lp_order:
        applied["linear_prediction"] = {
            "axis": "F1",
            "mode": "LPfr",
            "order": lp_order,
            "size": f1_points,
        }
    if y["baseline_mode"] or x["baseline_mode"]:
        applied["baseline"] = (y["baseline_mode"], x["baseline_mode"])
    return result, applied
