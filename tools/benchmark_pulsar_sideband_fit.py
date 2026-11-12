"""Time a repeatable PULSAR fit with two MAS sidebands on either side.

Run from a checkout with the plugin installed. The synthetic target uses a
finer powder mesh than the fit, a fixed Gaussian broadening and seeded noise.
"""

from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from spectrochempy_nmr.pulsar import _central_transition_profile

POINTS = 8192
WIDTH_HZ = 100e3
# Optimize in MHz and kHz so the three parameters are on comparable scales.
TRUE_PARAMETERS = (4.2, 0.55, 3.1)
INITIAL_PARAMETERS = (4.0, 0.48, 2.9)
BROADENING_SIGMA_HZ = 110


def broadened_profile(parameters, accuracy):
    """Calculate and broaden a normalized central-transition profile."""
    cq_mhz, eta, shift_khz = parameters
    _, profile = _central_transition_profile(
        cq_hz=cq_mhz * 1e6,
        eta=eta,
        shift_hz=shift_khz * 1e3,
        larmor_hz=100e6,
        spectral_width_hz=WIDTH_HZ,
        spinning_hz=8e3,
        sidebands=2,
        points=POINTS,
        accuracy=accuracy,
    )
    sigma_bins = BROADENING_SIGMA_HZ / (WIDTH_HZ / (POINTS - 1))
    return gaussian_filter1d(profile, sigma_bins, mode="constant")


def fit_profile(target, accuracy):
    """Fit the three physical parameters and count all residual evaluations."""
    calls = 0
    peak = target.max()

    def residual(parameters):
        nonlocal calls
        calls += 1
        return (broadened_profile(parameters, accuracy) - target) / peak

    broadened_profile(INITIAL_PARAMETERS, accuracy)  # Warm the powder mesh cache.
    start = perf_counter()
    result = least_squares(
        residual,
        INITIAL_PARAMETERS,
        bounds=((3.0, 0.05, 1.0), (5.0, 0.95, 5.0)),
        diff_step=1e-4,
    )
    elapsed = perf_counter() - start
    if not result.success:
        raise RuntimeError(result.message)
    return result, elapsed, calls


def main():
    """Compare fit speed and error at two powder mesh densities."""
    target = broadened_profile(TRUE_PARAMETERS, accuracy=32)
    rng = np.random.default_rng(82)
    target += rng.normal(0, 0.002 * target.max(), target.size)
    print(f"Target Cq (MHz), eta, shift (kHz): {TRUE_PARAMETERS}")
    for accuracy in (8, 16):
        result, elapsed, calls = fit_profile(target, accuracy)
        rms = np.sqrt(np.mean(result.fun**2))
        cq, eta, shift = result.x
        print(
            f"accuracy={accuracy}: {elapsed:.3f} s, {calls} evaluations, "
            f"Cq={cq:.6f} MHz, eta={eta:.6f}, shift={shift:.6f} kHz, "
            f"residual RMS={rms:.4%} of peak"
        )


if __name__ == "__main__":
    main()
