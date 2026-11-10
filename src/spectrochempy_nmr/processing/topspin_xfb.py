"""Replay the qualified uniform Bruker XFB profiles without prediction."""

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.units import ur
from spectrochempy_nmr.processing.channel_interop import _cross_pairs
from spectrochempy_nmr.processing.channel_interop import _rance_kay

_PROFILES = {
    "bruker-ser-qf-2d-v1": "qf",
    "bruker-ser-states-tppi-2d-v1": "states-tppi",
    "bruker-ser-echo-antiecho-2d-v1": "echo-antiecho",
}


def _check_axis(params, points, words_per_point, ft_mode, phase_mode):
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
        or params.get("BC_mod") != 0
        or params.get("WDW") not in (0, 4)
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
    if params["WDW"] == 4:
        shift = params.get("SSB")
        if not isinstance(shift, (int, float)) or not np.isfinite(shift) or shift < 0:
            raise ValueError("TopSpin XFB requires a valid squared sine window shift")


def _window(params, points, delay=0):
    weights = np.ones(points)
    if params["WDW"] == 4:
        shift = params["SSB"]
        phase = np.pi / shift if shift >= 2 else 0
        position = (np.arange(points) - delay) / (points - delay + 0.5)
        weights = np.sin(phase + (np.pi - phase) * position) ** 2
    weights[0] *= params["FCOR"]
    return weights


def _positive_fft(data, size, axis):
    return np.fft.fftshift(np.fft.ifft(data, n=size, axis=axis) * size, axes=axis)


def _processed_axis(params, source_axis):
    bins = np.arange(params["SI"]) / params["SI"]
    axis = Coord(
        params["OFFSET"] - bins * (params["SW_p"] / params["SF"]),
        units="ppm",
        title=source_axis.title,
        rounding=False,
        sigdigits=15,
    )
    axis.meta["reference_frequency"] = params["SF"] * ur.MHz
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
    if not qf and data.shape[0] != 2:
        raise ValueError("TopSpin XFB requires both acquired branches")
    ny, nx = data.shape[-2:]
    widths = acquisition["spectral_width_hz"]
    for axis, size, width in zip((source.y, source.x), (ny, nx), widths, strict=True):
        if axis.units != ur.s or not np.allclose(
            axis.data, np.arange(size) / width, rtol=1e-9, atol=0
        ):
            raise ValueError("TopSpin XFB requires the original calibrated time grids")

    status = meta.get("source_metadata", {})
    y, x = status.get("proc2s", {}), status.get("procs", {})
    if y.get("ME_mod") != 0 or x.get("ME_mod") != 0:
        raise ValueError("TopSpin XFB linear prediction is not supported")
    _check_axis(
        y, ny, 1 if qf else 2, 4 if mode == "states-tppi" else 6, 2 if qf else 1
    )
    _check_axis(x, nx, 2, 6, 0 if qf else 1)
    if y.get("PKNL") is not False or not isinstance(x.get("PKNL"), bool):
        raise ValueError("TopSpin XFB supports PKNL only on F2")
    delay = status.get("acqus", {}).get("GRPDLY")
    if (
        not isinstance(delay, (int, float))
        or not np.isfinite(delay)
        or not 0 <= delay < nx - 1
        or delay != int(delay)
    ):
        raise ValueError("TopSpin XFB requires a known integer direct group delay")

    f2_input = data * _window(x, nx, int(delay))
    if mode == "echo-antiecho":
        f2_input = _rance_kay(f2_input)
    sx, sy = x["SI"], y["SI"]
    f2 = _positive_fft(f2_input, sx, -1)
    phase_x = x["PHC0"] + (x["PHC1"] + 360 * delay * x["PKNL"]) * np.arange(sx) / sx
    f2 *= np.exp(-1j * np.deg2rad(phase_x))
    f1_window = _window(y, ny)[:, None]
    if qf:
        f1 = np.fft.fftshift(np.fft.fft(f2 * f1_window, n=sy, axis=-2), axes=-2)
        spectrum = np.abs(f1)
        rr = spectrum
    else:
        cross = _cross_pairs(f2) * f1_window
        if mode == "states-tppi":
            cross[:, 1::2] *= -1
        a, b = _positive_fft(cross, sy, -2)
        if mode == "states-tppi":
            rr_ir, ri_ii = a.real - 1j * b.real, -a.imag + 1j * b.imag
        else:
            rr_ir, ri_ii = a.imag - 1j * b.imag, a.real - 1j * b.real
        phase_y = y["PHC0"] + y["PHC1"] * np.arange(sy) / sy
        rotation = np.exp(1j * np.deg2rad(phase_y))[:, None]
        # Rotate F1 in pairs (2rr, 2ri) and (2ir, 2ii).
        real = (rr_ir.real + 1j * ri_ii.real) * rotation
        imag = (rr_ir.imag + 1j * ri_ii.imag) * rotation
        spectrum = np.stack((real.real + 1j * imag.real, real.imag + 1j * imag.imag))
        rr = spectrum[0].real

    cy, cx = sy // 2, sx // 2
    rr[cy, cx] = (rr[cy - 1, cx] + rr[cy + 1, cx]) / 2
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
    if not qf:
        result.meta.component_labels = ("S_R", "S_I")
        result.meta.quadrature_mode = "complex-channel-pair-2d"
    result.meta.readonly = readonly
    return result, {
        "replay": "topspin",
        "command": "xfb",
        "size": (sy, sx),
        "window": (y["WDW"], x["WDW"]),
        "group_delay_points": delay,
        "pknl": x["PKNL"],
        "phase": {
            "f1": (y["PHC0"], y["PHC1"]),
            "f2": (x["PHC0"], x["PHC1"]),
        },
        "axis_calibration": "procs",
    }
