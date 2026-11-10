"""Compare PBG1 band areas with a finite pulse and 3QMAS-derived sites."""

from argparse import ArgumentParser
from pathlib import Path

import nmrglue as ng
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import nnls

from spectrochempy_nmr.pulsar import IdealPulse
from spectrochempy_nmr.pulsar import Pulse
from spectrochempy_nmr.pulsar import simulate_pulse_sequence

from measure_pulsar_pbg_bands import ORDERS
from measure_pulsar_pbg_bands import ROTOR_HZ
from measure_pulsar_pbg_bands import measure_bands
from measure_pulsar_pbg_bands import read_pbg_spectrum

# Five regions in the 3QMAS map, in ppm (F1 lower/upper, F2 lower/upper).
# The bounds are defined from experiment 13 alone.
REGIONS = (
    (-14, 0, -22, 3),
    (1, 14, -8, 13),
    (37, 46, 32, 46),
    (44, 58, 41, 57),
    (63, 77, 62, 77),
)
RF_HZ = 50_000
WIDTHS_HZ = (150, 300, 600, 1200, 2400, 4800)


def ppm_axis(proc):
    return proc["OFFSET"] - np.arange(proc["SI"]) * (
        proc["SW_p"] / proc["SF"] / proc["SI"]
    )


def mqmas_regions(experiment):
    """Yield baseline-corrected 3QMAS patches and their ppm grids."""
    metadata, signal = ng.bruker.read_pdata(str(experiment / "pdata" / "1"))
    p1, p2 = metadata["proc2s"], metadata["procs"]
    if (
        signal.shape != (128, 1024)
        or metadata["acqus"]["PULPROG"] != "pf_3Qzf"
        or not np.isclose(p1["SF"] / p2["SF"], 17 / 12)
    ):
        raise ValueError("Expected the sheared PBG1 3QMAS spectrum, experiment 13.")
    f1, f2 = ppm_axis(p1), ppm_axis(p2)
    for f1_min, f1_max, f2_min, f2_max in REGIONS:
        rows = (f1 > f1_min) & (f1 < f1_max)
        columns = (f2 > f2_min) & (f2 < f2_max)
        background = (f2 > f2_max + 7) & (f2 < f2_max + 14)
        baseline = np.median(signal[np.ix_(rows, background)], axis=1)
        patch = np.maximum(signal[np.ix_(rows, columns)] - baseline[:, None], 0)
        yield f1[rows], f2[columns], patch, p2["SF"] * 1e6


def mqmas_parameters(experiment):
    """Estimate chemical shifts and Pq = |Cq| sqrt(1 + eta**2/3)."""
    parameters = []
    for f1, f2, patch, larmor_hz in mqmas_regions(experiment):
        iso_center = patch.sum(axis=1) @ f1 / patch.sum()
        mas_center = patch.sum(axis=0) @ f2 / patch.sum()

        # Spin 5/2: F2 = delta_iso + QIS and scaled F1 = delta_iso - 10/17 QIS.
        # QIS (ppm) = -0.006 (Pq / nu_L)**2 * 1e6.
        qis = (mas_center - iso_center) * 17 / 27
        if not np.isfinite(qis) or qis >= 0:
            raise ValueError("Expected a negative quadrupolar-induced shift.")
        pq_hz = larmor_hz * np.sqrt(-qis / 6000)
        parameters.append((mas_center - qis, pq_hz))
    return parameters


def simulate_components(sites, eta, acqu, proc, pulse, accuracy, sidebands):
    """Return complex profiles on a common grid in the experimental frame."""
    larmor_hz = proc["SF"] * 1e6
    carrier_hz = acqu["SFO1"] * 1e6 - larmor_hz
    components = []
    for shift_ppm, pq_hz in sites:
        spectrum = simulate_pulse_sequence(
            [pulse],
            spin=2.5,
            larmor_hz=larmor_hz,
            spectral_width_hz=4e6,
            cq_hz=pq_hz / np.sqrt(1 + eta**2 / 3),
            eta=eta,
            shift_hz=shift_ppm * larmor_hz * 1e-6 - carrier_hz,
            points=65537,
            accuracy=accuracy,
            spinning_hz=ROTOR_HZ,
            sidebands=sidebands,
            max_rf_step_s=0.125e-6,
        )
        axis = spectrum.x.data
        step_hz = axis[1] - axis[0]
        components.append(spectrum.data)
    return axis + step_hz / 2 + carrier_hz, np.column_stack(components)


def calibrate_center(frequency, observed, axis, components):
    """Choose one broadening and nonnegative site weights from the 1D center."""
    fit = np.abs(frequency) < 14_000
    scale = np.max(observed[fit])
    target = observed[fit] / scale
    step_hz = axis[1] - axis[0]
    best = None
    for width_hz in WIDTHS_HZ:
        sigma = width_hz / (np.sqrt(8 * np.log(2)) * step_hz)
        broad = gaussian_filter1d(components, sigma, axis=0)
        model = np.column_stack(
            [np.interp(frequency, axis, component) for component in broad.T]
        )
        central = model[fit]
        # Centering is equivalent to fitting a common additive baseline.
        means = central.mean(axis=0)
        weights, residual = nnls(central - means, target - target.mean())
        if best is None or residual < best[0]:
            best = residual, width_hz, weights, model @ weights
    residual, width_hz, weights, profile = best
    return residual / np.sqrt(fit.sum()), width_hz, weights / weights.sum(), profile


def ratios(frequency, signal):
    areas, _ = measure_bands(frequency, signal)
    center = areas[ORDERS.index(0), 1]
    if not np.isfinite(center) or center <= 0:
        raise ValueError("Expected positive area in the central window.")
    return 100 * areas[:, 1] / center


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("pbg1", type=Path, help="Path to the PBG1_2,5mm directory")
    parser.add_argument("--accuracy", type=int, default=8)
    parser.add_argument("--sidebands", type=int, default=45)
    args = parser.parse_args()
    if args.sidebands < 10:
        parser.error("--sidebands must include orders -10 through +10.")
    acqu, proc, frequency, observed = read_pbg_spectrum(args.pbg1 / "1")
    sites = mqmas_parameters(args.pbg1 / "13")
    for i, (shift_ppm, pq_hz) in enumerate(sites, 1):
        print(f"Region {i}: delta_iso={shift_ppm:.2f} ppm, Pq={pq_hz / 1e6:.3f} MHz")

    measured = ratios(frequency, observed)
    satellite = np.asarray(ORDERS) != 0
    modeled = {}
    for eta, pulse in (
        (0.0, Pulse(acqu["P"][1] * 1e-6, RF_HZ)),
        (0.5, Pulse(acqu["P"][1] * 1e-6, RF_HZ)),
        (1.0, Pulse(acqu["P"][1] * 1e-6, RF_HZ)),
        (0.5, IdealPulse(90)),
    ):
        kind = "ideal" if isinstance(pulse, IdealPulse) else "finite"
        axis, components = simulate_components(
            sites, eta, acqu, proc, pulse, args.accuracy, args.sidebands
        )
        rms, width, weights, profile = calibrate_center(
            frequency, observed, axis, components.real
        )
        predicted = ratios(frequency, profile)
        modeled[kind, eta] = predicted
        errors = np.abs(predicted[satellite] - measured[satellite])
        print(
            f"{kind}, eta={eta:g}: central RMS={rms:.2%}, FWHM={width} Hz, "
            f"weights={np.round(weights, 3)}, "
            f"area MAE/max={errors.mean():.3f}/{errors.max():.3f} pp, "
            f"summed bands={predicted[satellite].sum():.2f}%"
        )
    print(f"Measured summed bands={measured[satellite].sum():.2f}%")
    print("order,measured_pct,finite_eta0,finite_eta05,finite_eta1,ideal_eta05")
    for i, order in enumerate(ORDERS):
        predictions = ",".join(f"{values[i]:.3f}" for values in modeled.values())
        print(f"{order:+d},{measured[i]:.3f},{predictions}")


if __name__ == "__main__":
    main()
