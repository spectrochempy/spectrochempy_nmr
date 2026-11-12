"""Compare PULSAR satellite bands with the experimental DMFIT YAG spectrum."""

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter1d

from spectrochempy_nmr.pulsar import _central_transition_profile

from _pulsar_csdf import read_real_csdf

ROTOR_HZ = 15247.0679
FWHM_HZ = 316.008934
# Published MRSimulator fit: Cq in Hz, asymmetry and isotropic shift in ppm.
SITES = ((6066825.17, 1.668e-5, 77.4544449), (679126.785, 0.55123104, 0.91916103))


def model_components(frequency, larmor_hz, accuracy, sidebands):
    """Simulate on a wide axis, then interpolate onto the measured window."""
    components = []
    for cq_hz, eta, shift_ppm in SITES:
        axis, mass = _central_transition_profile(
            cq_hz=cq_hz,
            eta=eta,
            larmor_hz=larmor_hz,
            shift_hz=shift_ppm * larmor_hz * 1e-6,
            spectral_width_hz=8e6,
            points=65536,
            accuracy=accuracy,
            spinning_hz=ROTOR_HZ,
            spin=2.5,
            transitions="all",
            sidebands=sidebands,
        )
        step = axis[1] - axis[0]
        sigma_bins = FWHM_HZ / (np.sqrt(8 * np.log(2)) * step)
        broadened = gaussian_filter1d(mass, sigma_bins)
        components.append(np.interp(frequency, axis + step / 2, broadened))
    return np.column_stack(components)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("csdf", type=Path, help="Path to the original YAG CSDF")
    parser.add_argument("--accuracy", type=int, default=8)
    parser.add_argument("--sidebands", type=int, default=96)
    args = parser.parse_args()
    if args.sidebands < 8:
        parser.error(
            "--sidebands must be at least 8 for the measured comparison windows."
        )
    frequency, observed, larmor_hz = read_real_csdf(args.csdf)
    ppm = frequency * 1e6 / larmor_hz
    noise = (ppm > -570) & (ppm < -510)
    observed -= observed[noise].mean()

    start = perf_counter()
    components = model_components(frequency, larmor_hz, args.accuracy, args.sidebands)
    central = np.abs(frequency) < 10e3
    scale = np.linalg.lstsq(components[central], observed[central], rcond=None)[0]
    predicted = components @ scale
    print(f"Simulated and calibrated both sites in {perf_counter() - start:.3f} s")
    windows = []
    observed_areas, predicted_areas, peak_deltas = [], [], []
    for order in range(-8, 9):
        if order == 0:
            continue
        band = np.abs(frequency - order * ROTOR_HZ) < 3000
        windows.append(band)
        observed_areas.append(observed[band].sum())
        predicted_areas.append(predicted[band].sum())
        peak_deltas.append(
            abs(
                frequency[band][observed[band].argmax()]
                - frequency[band][predicted[band].argmax()]
            )
        )
    area_errors = np.abs(np.asarray(predicted_areas) / observed_areas - 1)
    bands = np.any(windows, axis=0)
    print(
        f"Pointwise sideband correlation: {np.corrcoef(observed[bands], predicted[bands])[0, 1]:.5f}"
    )
    print(
        f"Sideband area relative error (median / max): {np.median(area_errors):.2%} / {area_errors.max():.2%}"
    )
    print(
        f"Sideband peak difference (median / max): {np.median(peak_deltas):.1f} / {max(peak_deltas):.1f} Hz"
    )


if __name__ == "__main__":
    main()
