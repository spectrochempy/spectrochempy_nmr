"""Fit the processed Al(acac)3 central band from the MRSimulator tutorial."""

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import numpy as np
from nmrglue.fileio.bruker import read_pdata
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from spectrochempy_nmr.pulsar import _central_transition_profile


def read_central_band(pdata):
    """Read the phased Bruker spectrum on its processed frequency axis."""
    metadata, data = read_pdata(pdata)
    procs = metadata["procs"]
    larmor_mhz = procs["SF"]
    frequency = procs["OFFSET"] * larmor_mhz - (
        np.arange(procs["SI"]) * procs["SW_p"] / procs["SI"]
    )
    # The supplied pdata is zero-filled to 32768 points. Use every 16th
    # point on the increasing axis, then select the central-band window.
    frequency, data = frequency[::-16], data[::-16]
    mask = (frequency >= -1300) & (frequency <= 1100)
    return frequency[mask], data[mask] / data.max(), larmor_mhz * 1e6


def fit_band(frequency, observed, larmor_hz, spinning_hz, accuracy):
    """Fit the line shape, solving amplitude and linear baseline separately."""
    calls = 0
    constant = np.ones_like(frequency)
    slope = frequency / 1000

    def residual(parameters):
        nonlocal calls
        calls += 1
        cq_mhz, eta, shift_khz, sigma_hz = parameters
        axis, mass = _central_transition_profile(
            cq_hz=cq_mhz * 1e6,
            eta=eta,
            shift_hz=shift_khz * 1000,
            larmor_hz=larmor_hz,
            spectral_width_hz=12500,
            points=4096,
            spin=2.5,
            spinning_hz=spinning_hz,
            accuracy=accuracy,
        )
        step = axis[1] - axis[0]
        profile = gaussian_filter1d(mass, sigma_hz / step)
        profile = np.interp(frequency, axis + step / 2, profile)
        design = np.column_stack((profile, constant, slope))
        coefficients = np.linalg.lstsq(design, observed, rcond=None)[0]
        return design @ coefficients - observed

    initial = (3.0, 0.35, 0.9, 55)
    residual(initial)
    calls = 0
    start = perf_counter()
    result = least_squares(
        residual,
        initial,
        bounds=((1, 0.001, -0.3, 3), (5, 0.999, 2.0, 200)),
        diff_step=2e-4,
    )
    elapsed = perf_counter() - start
    if not result.success:
        raise RuntimeError(result.message)
    return result, elapsed, calls


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("pdata", type=Path, help="Path to the extracted pdata/1")
    parser.add_argument("--rotor-hz", type=float, default=12500)
    args = parser.parse_args()
    frequency, observed, larmor_hz = read_central_band(args.pdata)
    for accuracy in (8, 16, 32):
        fit, elapsed, calls = fit_band(
            frequency, observed, larmor_hz, args.rotor_hz, accuracy
        )
        cq, eta, shift, sigma = fit.x
        fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
        rms = np.sqrt(np.mean(fit.fun**2))
        print(
            f"accuracy={accuracy}: {elapsed:.3f} s, {calls} evaluations, "
            f"Cq={cq:.6f} MHz, eta={eta:.6f}, shift={shift * 1000:.1f} Hz, "
            f"Gaussian FWHM={fwhm:.1f} Hz, residual RMS={rms:.2%} of peak"
        )


if __name__ == "__main__":
    main()
