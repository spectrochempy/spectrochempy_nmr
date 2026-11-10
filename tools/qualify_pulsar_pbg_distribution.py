"""Test joint shift and quadrupolar spreads inferred from PBG1 3QMAS."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from spectrochempy_nmr.pulsar import Pulse

from measure_pulsar_pbg_bands import ORDERS
from qualify_pulsar_pbg import RF_HZ
from qualify_pulsar_pbg import calibrate_center
from qualify_pulsar_pbg import mqmas_regions
from qualify_pulsar_pbg import ratios
from qualify_pulsar_pbg import simulate_components
from qualify_pulsar_pbg_fid import read_pbg_fid
from qualify_pulsar_pbg_fid import replay_efp
from qualify_pulsar_pbg_fid import synthesize_fid

ETA = 0.5


def ridge_nodes(f1, f2, patch, larmor_hz, count, minimum_fraction=0.0):
    """Compress the positive 3QMAS patch into equal-intensity F1 slices."""
    mass = patch.sum(axis=1)
    keep = mass > 0
    mass = mass[keep]
    if not mass.size:
        raise ValueError("Expected a positive 3QMAS region.")
    f2_centers = (patch[keep] @ f2) / mass
    row_edges = np.r_[0, mass.cumsum() / mass.sum()]
    edges = np.linspace(0, 1, count + 1)
    # Fractional boundary rows keep the regional first moments unchanged.
    overlap = np.maximum(
        0,
        np.minimum(edges[1:, None], row_edges[None, 1:])
        - np.maximum(edges[:-1, None], row_edges[None, :-1]),
    )
    weights = overlap / overlap.sum(axis=1, keepdims=True)
    iso = weights @ f1[keep]
    mas = weights @ f2_centers
    qis = (mas - iso) * 17 / 27
    shift = mas - qis
    mean_qis = qis.mean()
    if not np.isfinite(mean_qis) or mean_qis >= 0:
        raise ValueError("Expected a negative mean quadrupolar-induced shift.")
    # Shrink deviations uniformly if a weak slice would imply negative Pq².
    # This preserves the regional means of shift and Pq².
    shrink = 1.0
    limit = minimum_fraction * mean_qis
    if qis.max() > limit:
        shrink = (limit - mean_qis) / (qis.max() - mean_qis)
    qis = mean_qis + shrink * (qis - mean_qis)
    pq_hz = larmor_hz * np.sqrt(np.maximum(-qis, 0) / 6000)
    return list(zip(shift, pq_hz, strict=True)), shrink


def compare(pbg1, accuracy, sidebands, slices, rf_hz, minimum_fraction):
    acqu, proc, frequency, published, fid = read_pbg_fid(pbg1 / "1")
    observed = replay_efp(fid, acqu, proc).real
    measured = ratios(frequency, observed)
    pulse = Pulse(acqu["P"][1] * 1e-6, rf_hz)
    regions = list(mqmas_regions(pbg1 / "13"))
    satellite = np.asarray(ORDERS) != 0
    columns = [measured]
    print(
        f"RF = {rf_hz / 1000:.3f} kHz. Measured sum: {measured[satellite].sum():.3f}%",
        flush=True,
    )
    for count in (1, slices):
        groups = []
        for index, region in enumerate(regions, 1):
            sites, shrink = ridge_nodes(*region, count, minimum_fraction)
            nodes = ", ".join(f"({s:.3f}, {p / 1e6:.3f})" for s, p in sites)
            print(
                f"{count} slices, region {index}: "
                f"(ppm, MHz)={nodes}; "
                f"QIS spread scale={shrink:.3f}",
                flush=True,
            )
            axis, spectra = simulate_components(
                sites, ETA, acqu, proc, pulse, accuracy, sidebands
            )
            groups.append(spectra.mean(axis=1))
        components = np.column_stack(groups)
        # Keep the existing comparison protocol. Fit only the published 1D
        # center, then apply the same EFP and local baselines to both FIDs.
        rms, width, weights, _ = calibrate_center(
            frequency, published, axis, components.real
        )
        synthetic = synthesize_fid(
            axis, components @ weights, width, acqu, proc, acqu["DE"] * 1e-6
        )
        predicted = ratios(frequency, replay_efp(synthetic, acqu, proc).real)
        errors = np.abs(predicted[satellite] - measured[satellite])
        print(
            f"{count} slices: central RMS={rms:.2%}, FWHM={width} Hz, "
            f"weights={np.round(weights, 3)}, "
            f"sum={predicted[satellite].sum():.3f}%, "
            f"area MAE/max={errors.mean():.3f}/{errors.max():.3f} pp",
            flush=True,
        )
        columns.append(predicted)
    print("order,measured_pct,point_pct,distributed_pct")
    for order, values in zip(ORDERS, zip(*columns, strict=True), strict=True):
        print(f"{order:+d}," + ",".join(f"{value:.3f}" for value in values))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("pbg1", type=Path, help="Path to the PBG1_2,5mm directory")
    parser.add_argument("--accuracy", type=int, default=4)
    parser.add_argument("--sidebands", type=int, default=20)
    parser.add_argument("--slices", type=int, default=3)
    parser.add_argument("--rf-hz", type=float, default=RF_HZ)
    parser.add_argument(
        "--minimum-qis-fraction",
        type=float,
        default=0.0,
        help="Minimum absolute slice QIS as a fraction of the regional mean.",
    )
    args = parser.parse_args()
    if args.sidebands < 10 or args.slices < 2:
        parser.error("Use at least 10 sideband orders and 2 slices.")
    if not np.isfinite(args.rf_hz) or args.rf_hz <= 0:
        parser.error("--rf-hz must be finite and positive.")
    if not 0 <= args.minimum_qis_fraction < 1:
        parser.error("--minimum-qis-fraction must be in [0, 1).")
    compare(
        args.pbg1,
        args.accuracy,
        args.sidebands,
        args.slices,
        args.rf_hz,
        args.minimum_qis_fraction,
    )


if __name__ == "__main__":
    main()
