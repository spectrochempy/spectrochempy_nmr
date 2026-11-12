"""Measure reference sideband areas in the public PBG1 27Al MAS spectrum."""

from argparse import ArgumentParser
from pathlib import Path

import nmrglue as ng
import numpy as np
from numpy.polynomial import Polynomial

ROTOR_HZ = 30_000
ORDERS = range(-10, 11)
HALF_WIDTHS_HZ = (4500, 6000, 7500)


def measure_bands(frequency, signal):
    """Return integrated areas for three windows and the peak position per order."""
    bin_hz = abs(frequency[1] - frequency[0])
    areas = []
    peaks = []
    for order in ORDERS:
        offset = frequency - order * ROTOR_HZ
        # Keep the baseline windows separate from the integration windows and
        # within the interval assigned to this rotor order.
        baseline = (abs(offset) >= 8500) & (abs(offset) <= 12_500)
        corrected = signal - Polynomial.fit(offset[baseline], signal[baseline], 1)(
            offset
        )
        areas.append(
            [corrected[abs(offset) <= width].sum() * bin_hz for width in HALF_WIDTHS_HZ]
        )
        main_window = abs(offset) <= HALF_WIDTHS_HZ[1]
        peaks.append(frequency[main_window][corrected[main_window].argmax()])
    return np.asarray(areas), np.asarray(peaks)


def read_pbg_spectrum(experiment):
    """Return the original 1D acquisition metadata, grid and signal."""
    metadata, signal = ng.bruker.read_pdata(str(experiment / "pdata" / "1"))
    proc, acqu = metadata["procs"], metadata["acqus"]
    if (
        acqu["NUC1"] != "27Al"
        or acqu["PULPROG"] != "pf_zg"
        or acqu["MASR"] != ROTOR_HZ
        or acqu["P"][1] != 0.5
        or acqu["PLW"][1] != 32
        or not np.isclose(acqu["SFO1"], 195.439811, rtol=0, atol=1e-6)
        or proc["SW_p"] != 2_500_000
        or signal.shape != (16_384,)
    ):
        raise ValueError("Expected the original PBG1 27Al spectrum, experiment 1.")

    # TopSpin OFFSET is the left edge in ppm, and SF is in MHz.
    frequency = proc["OFFSET"] * proc["SF"] - np.arange(signal.size) * (
        proc["SW_p"] / signal.size
    )
    return acqu, proc, frequency, signal


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path, help="Path to PBG1_2,5mm/1")
    args = parser.parse_args()
    try:
        _, _, frequency, signal = read_pbg_spectrum(args.experiment)
    except ValueError as error:
        parser.error(str(error))
    areas, peaks = measure_bands(frequency, signal)
    central = areas[ORDERS.index(0)]
    if not np.all(np.isfinite(areas)) or np.any(central <= 0):
        parser.error("Expected finite band areas and a positive central area.")
    ratios = 100 * areas / central
    shifts = peaks - np.asarray(ORDERS) * ROTOR_HZ
    print("order,peak_hz,peak_minus_order_hz,area_pct_4p5k,area_pct_6k,area_pct_7p5k")
    for order, peak, shift, row in zip(ORDERS, peaks, shifts, ratios, strict=True):
        print(
            f"{order:+d},{peak:.1f},{shift:.1f},{row[0]:.3f},{row[1]:.3f},{row[2]:.3f}"
        )


if __name__ == "__main__":
    main()
