"""Check CSA sideband areas on the experimental DMFIT 31P MAS spectrum."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from spectrochempy_nmr.pulsar import simulate_csa

from _pulsar_csdf import read_real_csdf

ROTOR_HZ = 6012.45932
SHIFT_PPM = 7.58326133
SHIELDING_ZETA_PPM = -88.1880920
ETA = 0.45591094
ORDERS = np.arange(-3, 4)


def band_areas(frequency, data, shift_hz):
    """Integrate and normalize seven resolved rotor orders."""
    centers = shift_hz + ORDERS * ROTOR_HZ
    included = np.abs(frequency[:, None] - centers) < 1400
    areas = data @ included
    return areas / areas.sum()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("csdf", type=Path, help="Path to the original phosphate CSDF")
    args = parser.parse_args()
    frequency, observed, larmor_hz = read_real_csdf(args.csdf)
    hz_per_ppm = larmor_hz * 1e-6
    shift = SHIFT_PPM * hz_per_ppm
    observed -= observed[frequency / hz_per_ppm < -100].mean()
    measured = band_areas(frequency, observed, shift)
    # The published tensor is a shielding tensor. PULSAR takes chemical
    # shift anisotropy, with the opposite sign.
    anisotropy = -SHIELDING_ZETA_PPM * hz_per_ppm
    for label, sign in (("chemical-shift sign", 1), ("reversed sign", -1)):
        spectrum = simulate_csa(
            anisotropy_hz=sign * anisotropy,
            eta=ETA,
            shift_hz=shift,
            spinning_hz=ROTOR_HZ,
            sidebands=3,
            spectral_width_hz=60e3,
        )
        predicted = band_areas(spectrum.x.data, spectrum.data, shift)
        error = np.abs(measured - predicted)
        print(
            f"{label}: maximum area difference={error.max():.4%}, L1={error.sum():.4%}"
        )


if __name__ == "__main__":
    main()
