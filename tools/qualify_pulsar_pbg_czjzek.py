"""Predict PBG1 sideband areas from 3QMAS-constrained Czjzek ensembles."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from spectrochempy_nmr.czjzek import CzjzekDistribution
from spectrochempy_nmr.czjzek import simulate_czjzek
from spectrochempy_nmr.pulsar import Pulse

from measure_pulsar_pbg_bands import ORDERS
from measure_pulsar_pbg_bands import ROTOR_HZ
from qualify_pulsar_pbg import RF_HZ
from qualify_pulsar_pbg import calibrate_center
from qualify_pulsar_pbg import mqmas_regions
from qualify_pulsar_pbg import ratios
from qualify_pulsar_pbg_fid import read_pbg_fid
from qualify_pulsar_pbg_fid import replay_efp
from qualify_pulsar_pbg_fid import synthesize_fid


def constrain_region(f1, f2, patch, larmor_hz, carrier_hz, eta, local_floor):
    """Match F1 variance and F1/F2 covariance under independent disorder."""
    mass = patch / patch.sum()
    marginal = mass.sum(axis=1)
    mean_f1 = marginal @ f1
    mean_f2 = mass.sum(axis=0) @ f2
    x, y = f1 - mean_f1, f2 - mean_f2
    var_f1 = marginal @ (x * x)
    covariance = x @ mass @ y

    # F1 = shift + a*Y, F2 = shift - Y + orientation residual, where
    # Y = -QIS. The F2 residual is assumed to have zero conditional mean.
    a = 10 / 17
    mean_y = (mean_f1 - mean_f2) / (1 + a)
    if not np.isfinite(mean_y) or mean_y <= 0:
        raise ValueError("Expected a positive mean absolute QIS.")
    var_y = (var_f1 - covariance) / (a * (1 + a))
    relative_var = var_y / mean_y**2

    # Pq² = (2/3)||EFG||². For a noncentral chi-square with five degrees
    # of freedom, Var(Y)/E(Y)² = (2/5)*(1 - local_fraction²).
    # Project onto the feasible variance interval at fixed mean Y and
    # keep the observed F1 variance when allocating shift broadening.
    max_relative_var = min(0.4 * (1 - local_floor**2), var_f1 / (a * mean_y) ** 2)
    fitted_relative_var = np.clip(relative_var, 0, max_relative_var)
    local_fraction = np.sqrt(1 - 2.5 * fitted_relative_var)
    var_shift = max(var_f1 - a**2 * mean_y**2 * fitted_relative_var, 0)
    mean_pq_sq = larmor_hz**2 * mean_y / 6000
    hz_per_ppm = larmor_hz * 1e-6
    distribution = CzjzekDistribution(
        sigma_hz=np.sqrt(0.3 * mean_pq_sq * (1 - local_fraction)),
        cq_hz=np.sqrt(mean_pq_sq * local_fraction / (1 + eta**2 / 3)),
        eta=eta,
        shift_hz=(mean_f2 + mean_y) * hz_per_ppm - carrier_hz,
        shift_sigma_hz=np.sqrt(var_shift) * hz_per_ppm,
    )
    return distribution, relative_var, fitted_relative_var


def compare(pbg1, accuracy, sidebands, samples, seed, rf_hz, eta, local_floor):
    acqu, proc, frequency, published, fid = read_pbg_fid(pbg1 / "1")
    larmor_hz = proc["SF"] * 1e6
    carrier_hz = (acqu["SFO1"] - proc["SF"]) * 1e6
    pulse = Pulse(acqu["P"][1] * 1e-6, rf_hz)
    components = []
    print(
        f"samples={samples}, seed={seed}, accuracy={accuracy}, sidebands={sidebands}, "
        f"RF={rf_hz:g} Hz, eta0={eta:g}, local floor={local_floor:g}",
        flush=True,
    )
    for index, region in enumerate(mqmas_regions(pbg1 / "13"), 1):
        distribution, relative_var, fitted_var = constrain_region(
            *region, carrier_hz, eta, local_floor
        )
        print(
            f"region {index}: relative QIS variance observed/model "
            f"{relative_var:.3f}/{fitted_var:.3f}; "
            f"Cq0={distribution.cq_hz / 1e6:.3f} MHz; "
            f"EFG sigma={distribution.sigma_hz / 1e6:.3f} MHz; "
            f"shift sigma={distribution.shift_sigma_hz / (larmor_hz * 1e-6):.3f} ppm",
            flush=True,
        )
        spectrum = simulate_czjzek(
            distribution,
            spin=2.5,
            larmor_hz=larmor_hz,
            spectral_width_hz=4e6,
            samples=samples,
            seed=seed + index - 1,
            points=65537,
            accuracy=accuracy,
            spinning_hz=ROTOR_HZ,
            sidebands=sidebands,
            transitions="all",
            events=[pulse],
            max_rf_step_s=0.125e-6,
        )
        components.append(spectrum.data)
    axis = spectrum.x.data
    axis = axis + (axis[1] - axis[0]) / 2 + carrier_hz
    components = np.column_stack(components)
    # Replicate the previous benchmark's central-only calibration and EFP.
    rms, width, weights, _ = calibrate_center(
        frequency, published, axis, components.real
    )
    synthetic = synthesize_fid(
        axis, components @ weights, width, acqu, proc, acqu["DE"] * 1e-6
    )
    predicted = ratios(frequency, replay_efp(synthetic, acqu, proc).real)
    measured = ratios(frequency, replay_efp(fid, acqu, proc).real)
    satellite = np.asarray(ORDERS) != 0
    errors = np.abs(predicted[satellite] - measured[satellite])
    print(
        f"central RMS={rms:.2%}, FWHM={width} Hz, weights={np.round(weights, 3)}, "
        f"measured/model sum={measured[satellite].sum():.3f}/{predicted[satellite].sum():.3f}%, "
        f"area MAE/max={errors.mean():.3f}/{errors.max():.3f} pp",
        flush=True,
    )
    print("order,measured_pct,czjzek_pct")
    for order, data, model in zip(ORDERS, measured, predicted, strict=True):
        print(f"{order:+d},{data:.3f},{model:.3f}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("pbg1", type=Path, help="Path to the PBG1_2,5mm directory")
    parser.add_argument("--accuracy", type=int, default=4)
    parser.add_argument("--sidebands", type=int, default=20)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rf-hz", type=float, default=RF_HZ)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--local-floor", type=float, default=0.0)
    args = parser.parse_args()
    if args.sidebands < 10:
        parser.error("Use at least 10 sideband orders.")
    if not np.isfinite(args.rf_hz) or args.rf_hz <= 0:
        parser.error("--rf-hz must be finite and positive.")
    if not 0 <= args.local_floor <= 1:
        parser.error("--local-floor must be in [0, 1].")
    compare(**vars(args))


if __name__ == "__main__":
    main()
