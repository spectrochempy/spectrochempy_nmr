"""Replay TopSpin EFP on a digitally filtered, complex DQD acquisition."""

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.units import ur
from spectrochempy_nmr.processing.fid_preparation import validated_fid_inputs


def replay_dqd_efp(source):
    meta = source.meta
    if (
        source.ndim != 1
        or meta.get("data_domain") != "time"
        or meta.get("normalization_profile") != "bruker-fid-1d-complex-v1"
        or meta.get("aq_mod") != [3]
    ):
        raise ValueError("TopSpin replay requires a raw Bruker DQD 1D FID")

    acquisition, recipe, samples = validated_fid_inputs(source)
    if acquisition["digital_filter"]["applied"] or acquisition["dc_offset"]["applied"]:
        raise ValueError(
            "TopSpin replay requires remove_digital_filter=False and remove_dc_offset=False"
        )
    if samples.size != acquisition["storage_points"]:
        raise ValueError("TopSpin replay requires the full stored FID")

    profile = meta.nmr_processing.get("vendor_profile", {})
    pars = profile.get("parameters", {})
    if (
        pars.get("WDW") != 1
        or pars.get("PH_mod") != 1
        or pars.get("BC_mod") != 0
        or pars.get("ME_mod") != 0
        or pars.get("PKNL") is not True
        or pars.get("REVERSE") is not False
        or pars.get("FCOR") != 1
    ):
        raise ValueError(
            "TopSpin replay requires EFP status with PKNL enabled, FCOR=1, "
            "and no baseline correction, linear prediction or reversal"
        )

    size = pars.get("SI")
    effective_td = pars.get("TDeff")
    if (
        not isinstance(size, int)
        or size < 2
        or size % 2
        or pars.get("FTSIZE") != size
        or pars.get("STSI") != size
        or pars.get("STSR") != 0
        or not isinstance(effective_td, int)
        or not 0 < effective_td <= 2 * acquisition["acquired_points"]
        or effective_td % 2
    ):
        raise ValueError(
            "TopSpin replay requires full-spectrum EFP status with valid SI and TDeff"
        )

    filter_spec = recipe.get("digital_filter", {})
    if filter_spec.get("state") != "available":
        raise ValueError("TopSpin replay requires a known Bruker group delay")

    delay = filter_spec["group_delay_points"]
    origin = int(np.floor(delay + 0.5))
    points = min(size, effective_td // 2)
    if not 0 < delay < points - 1 or points > samples.size:
        raise ValueError("TopSpin replay requires enough unmodified acquired points")

    try:
        lb = float(pars["LB"].to("Hz").m)
        phase_zero = float(pars["PHC0"].to("degree").m)
        phase_first = float(pars["PHC1"].to("degree").m)
        width, frequency, offset = (
            float(pars[key]) for key in ("SW_p", "SF", "OFFSET")
        )
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "TopSpin replay requires valid EFP window, phase and axis parameters"
        ) from exc
    if (
        not np.isfinite([lb, phase_zero, phase_first, width, frequency, offset]).all()
        or min(width, frequency) <= 0
    ):
        raise ValueError(
            "TopSpin replay requires finite EFP window, phase and axis parameters"
        )

    time = (np.arange(points) - origin) / acquisition["spectral_width_hz"]
    windowed = samples[:points] * np.exp(-np.pi * lb * time)
    bins = np.arange(size) / size
    # Keep the fractional part in the delay ramp. The rounded delay sets
    # the apodization origin and the parity correction for the centered FFT.
    delay_phase = 360 * (delay * bins - origin / 2)
    phase = phase_zero + phase_first * bins + delay_phase
    spectrum = np.fft.fftshift(np.fft.fft(windowed, n=size))
    spectrum *= np.exp(1j * np.deg2rad(phase))
    if not np.isfinite(spectrum).all():
        raise ValueError("TopSpin replay produced nonfinite spectral values")

    coord = Coord(
        offset - bins * (width / frequency),
        units="ppm",
        title="chemical shift",
        rounding=False,
        sigdigits=15,
    )
    coord.meta["reference_frequency"] = frequency * ur.MHz
    coord.meta["acquisition_frequency"] = (
        acquisition["observation_frequency_mhz"] * ur.MHz
    )
    result = source.copy()
    result.delete_coordset()
    result.data = spectrum
    result.set_coordset(x=coord)
    readonly = result.meta.readonly
    result.meta.readonly = False
    result.meta.data_domain = "frequency"
    result.meta.isfreq = [True]
    result.meta.si = [size]
    result.meta.phased = [True]
    result.meta.phc0 = [phase_zero * ur.degree]
    result.meta.phc1 = [phase_first * ur.degree]
    result.meta.pivot = [offset * ur.ppm]
    result.meta.readonly = readonly
    result.history = "Replayed TopSpin DQD 1D EFP from procs"
    return result, {
        "replay": "topspin",
        "apodization": "em",
        "lb_hz": lb,
        "size": size,
        "group_delay_points": delay,
        "phase": {"phc0_deg": phase_zero, "phc1_deg": phase_first},
        "axis_calibration": "procs",
    }
