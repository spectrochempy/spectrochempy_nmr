"""Replay PBG1 TopSpin processing on the recorded and modeled FIDs."""

from argparse import ArgumentParser
from pathlib import Path

import nmrglue as ng
import numpy as np
from scipy.signal import zoom_fft

from spectrochempy_nmr.pulsar import Pulse

from measure_pulsar_pbg_bands import ORDERS
from measure_pulsar_pbg_bands import read_pbg_spectrum
from qualify_pulsar_pbg import RF_HZ
from qualify_pulsar_pbg import calibrate_center
from qualify_pulsar_pbg import mqmas_parameters
from qualify_pulsar_pbg import ratios
from qualify_pulsar_pbg import simulate_components


def replay_efp(fid, acqu, proc):
    """Reproduce the PBG1 complex TopSpin efp spectrum up to a real scale."""
    dwell_s = 1 / acqu["SW_h"]
    size = proc["SI"]
    fraction = np.arange(size) / size
    # TopSpin retains the digital filter lead-in during exponential weighting.
    window = np.exp(-np.pi * proc["LB"] * dwell_s * np.arange(fid.size))
    transformed = np.fft.fftshift(np.fft.fft((fid * window).conj(), size))
    delay_phase = 360 * acqu["GRPDLY"] * (fraction - 0.5)
    receiver_phase = proc["PHC0"] + proc["PHC1"] * fraction
    return transformed * np.exp(1j * np.deg2rad(delay_phase + receiver_phase))


def synthesize_fid(axis, profile, width_hz, acqu, proc, dead_time_s):
    """Sample modeled complex lines in the Bruker receiver convention."""
    sw_hz = acqu["SW_h"]
    delay = int(acqu["GRPDLY"])
    count = acqu["TD"] // 2 - delay
    times = dead_time_s + np.arange(count) / sw_hz
    carrier_hz = (acqu["SFO1"] - proc["SF"]) * 1e6
    # Recover the simulator's uniform bin grid from rounded coordinates.
    step_hz = (axis[-1] - axis[0]) / (axis.size - 1)
    offsets = axis[0] - carrier_hz + step_hz * np.arange(axis.size)
    # Use a unity passband inside Nyquist. The observed filter kernel is
    # unavailable, so the synthetic lead-in represents its delay alone.
    passband = np.abs(offsets) < sw_hz / 2
    offsets = offsets[passband]
    profile = profile[passband]
    # The zoom FFT sums the evenly spaced line masses at the acquired times.
    analytic = zoom_fft(
        profile, [dead_time_s, dead_time_s + count / sw_hz], m=count, fs=1 / step_hz
    )
    analytic *= np.exp(-2j * np.pi * offsets[0] * times)
    sigma_hz = width_hz / np.sqrt(8 * np.log(2))
    analytic *= np.exp(-2 * (np.pi * sigma_hz * times) ** 2)
    # Match the receiver's zero-order phase at the carrier. The first-order
    # phase comes from physical sampling after the specified dead time.
    carrier_phase = proc["PHC0"] + proc["PHC1"] / 2
    analytic *= np.exp(-1j * np.deg2rad(carrier_phase))
    return np.pad(analytic.conj(), (delay, 0))


def compare(pbg1, accuracy, sidebands):
    acqu, proc, frequency, published = read_pbg_spectrum(pbg1 / "1")
    _, fid = ng.bruker.read(str(pbg1 / "1"), read_pulseprogram=False)
    if (
        acqu["GRPDLY"] != 68
        or acqu["DIGMOD"] != 1
        or acqu["TD"] != 10_000
        or proc["SI"] != 16_384
        or proc["WDW"] != 1
        or proc["FT_mod"] != 6
        or proc["FCOR"] != 1
        or proc["PKNL"] is not True
        or fid.ndim != 1
        or fid.size < acqu["TD"] // 2
    ):
        raise ValueError("Expected the original PBG1 FID and efp parameters.")
    raw = replay_efp(fid[: acqu["TD"] // 2], acqu, proc).real
    archived, replayed = ratios(frequency, published), ratios(frequency, raw)
    satellite = np.asarray(ORDERS) != 0
    discrepancy = np.abs(archived[satellite] - replayed[satellite])
    print(
        f"Archived/replayed sums: {archived[satellite].sum():.3f}% / "
        f"{replayed[satellite].sum():.3f}%; area MAE/max: "
        f"{discrepancy.mean():.3f}/{discrepancy.max():.3f} pp",
        flush=True,
    )
    sites = mqmas_parameters(pbg1 / "13")
    pulse = Pulse(acqu["P"][1] * 1e-6, RF_HZ)
    dead_time_s = acqu["DE"] * 1e-6
    # ACQT0 is used by baseopt, whereas PBG1 used standard digital filtering.
    # Include its offset separately to quantify sensitivity to the time origin.
    adjusted_time_s = dead_time_s + acqu["ACQT0"] * 1e-6
    print(
        f"Synthetic acquisition starts at {dead_time_s * 1e6:.3f} µs (DE); "
        f"timing sensitivity at {adjusted_time_s * 1e6:.3f} µs (DE + ACQT0)"
    )
    columns = [archived, replayed]
    labels = ["order", "archived_pct", "raw_efp_pct"]
    for eta in (0.0, 0.5, 1.0):
        axis, components = simulate_components(
            sites, eta, acqu, proc, pulse, accuracy, sidebands
        )
        _, width_hz, weights, direct = calibrate_center(
            frequency, published, axis, components.real
        )
        # Keep the previous central fit fixed across the synthetic treatments.
        before = ratios(frequency, direct)
        profile = components @ weights
        after, adjusted = (
            ratios(
                frequency,
                replay_efp(
                    synthesize_fid(axis, profile, width_hz, acqu, proc, start_s),
                    acqu,
                    proc,
                ).real,
            )
            for start_s in (dead_time_s, adjusted_time_s)
        )
        difference = np.abs(after[satellite] - replayed[satellite])
        print(
            f"eta={eta:g}: simulation direct/efp-DE/efp-adjusted sums "
            f"{before[satellite].sum():.3f}% / {after[satellite].sum():.3f}% / "
            f"{adjusted[satellite].sum():.3f}%; "
            f"replay-matched area MAE/max {difference.mean():.3f}/"
            f"{difference.max():.3f} pp",
            flush=True,
        )
        labels.extend(
            [f"direct_eta{eta:g}", f"efp_de_eta{eta:g}", f"efp_adjusted_eta{eta:g}"]
        )
        columns.extend([before, after, adjusted])
    print(",".join(labels))
    for order, values in zip(ORDERS, zip(*columns, strict=True), strict=True):
        print(f"{order:+d}," + ",".join(f"{value:.3f}" for value in values))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("pbg1", type=Path, help="Path to the PBG1_2,5mm directory")
    parser.add_argument("--accuracy", type=int, default=8)
    parser.add_argument("--sidebands", type=int, default=45)
    args = parser.parse_args()
    if args.sidebands < 10:
        parser.error("--sidebands must include orders -10 through +10.")
    compare(args.pbg1, args.accuracy, args.sidebands)


if __name__ == "__main__":
    main()
