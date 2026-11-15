"""Compare PULSAR with MRSimulator 1.0.0 or SIMPSON 4.2.1.

Set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1
before running. Install the selected reference separately:
python tools/benchmark_pulsar_reference.py mrsimulator
python tools/benchmark_pulsar_reference.py simpson
"""

import argparse
import subprocess
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from spectrochempy.nmr import pulsar

SIMPSON_INPUT = """\
spinsys {
    channels 1H 13C
    nuclei 1H 13C
    jcoupling 1 2 128 0 0 0 0 0
}
par {
    proton_frequency 400e6
    spin_rate 0
    sw 2048
    np 1024
    crystal_file rep10
    gamma_angles 1
    start_operator I1x
    detect_operator I1p
}
proc pulseq {} {
    global par
    acq_block { delay [expr {1e6/$par(sw)}] }
}
proc main {} {
    set f [fsimpson]
    fsave $f j_doublet.fid
}
"""


def timed(simulate):
    simulate()
    times = []
    for _ in range(7):
        start = perf_counter()
        result = simulate()
        times.append((perf_counter() - start) * 1000)
    return result, median(times)


def align(left, mass, grid):
    step = left[1] - left[0]
    edges = np.append(left, left[-1] + step)
    cumulative = np.concatenate(([0.0], np.cumsum(mass) / mass.sum()))
    return np.diff(np.interp(grid, edges, cumulative))


def compare_mrsimulator():
    from mrsimulator import Simulator, Site, SpinSystem
    from mrsimulator.method.lib import BlochDecayCTSpectrum
    from mrsimulator.spin_system.isotope import Isotope

    isotope = Isotope(symbol="27Al")
    field = isotope.ref_freq_to_B0(104.38311571e6)
    larmor = abs(isotope.gyromagnetic_ratio * field * 1e6)
    angle = np.arccos(1 / np.sqrt(3))
    site = Site(isotope="27Al", quadrupolar={"Cq": 2.923e6, "eta": 0.136})
    width, points = 32_000, 16_384
    grid = np.linspace(-width / 2, width / 2, points + 1)
    for rate in (500, 1000):
        method = BlochDecayCTSpectrum(
            channels=["27Al"],
            magnetic_flux_density=field,
            rotor_frequency=rate,
            rotor_angle=angle,
            spectral_dimensions=[{"count": points, "spectral_width": width}],
        )
        simulator = Simulator(spin_systems=[SpinSystem(sites=[site])], methods=[method])
        simulator.config.integration_density = 320
        simulator.config.number_of_sidebands = 32
        simulator.config.number_of_gamma_angles = 1

        def simulate_reference():
            simulator.run(n_jobs=1)
            return simulator.methods[0].simulation

        actual, pulsar_ms = timed(
            lambda: pulsar.simulate_central_transition(
                spin=2.5,
                cq_hz=2.923e6,
                eta=0.136,
                larmor_hz=larmor,
                spinning_hz=rate,
                rotor_angle_deg=np.rad2deg(angle),
                spectral_width_hz=width,
                points=points,
                accuracy=64,
                sidebands=14,
            )
        )
        reference, reference_ms = timed(simulate_reference)
        dimension = reference.dimensions[0]
        dimension.to("Hz")
        frequency = dimension.coordinates.to_value("Hz")
        step = frequency[1] - frequency[0]
        p = align(actual.x.data, actual.data, grid)
        m = align(frequency - step / 2, reference.y[0].components[0].real, grid)
        difference = 100 * np.max(np.abs(p - m)) / m.max()
        print(
            f"{rate} Hz: PULSAR {pulsar_ms:.1f} ms, MRSimulator {reference_ms:.1f} ms"
        )
        print(f"Maximum bin difference: {difference:.6f}% of reference peak")


def compare_simpson():
    with TemporaryDirectory() as directory:
        path = Path(directory)
        (path / "j_doublet.in").write_text(SIMPSON_INPUT)
        subprocess.run(["simpson", "j_doublet.in"], cwd=path, check=True)
        fid = np.loadtxt(path / "j_doublet.fid", skiprows=5, max_rows=1024)
    frequency = np.fft.fftfreq(len(fid), d=1 / 2048)
    reference = np.abs(np.fft.fft(fid[:, 0] + 1j * fid[:, 1]))
    actual = pulsar.simulate_coupled_sequence(
        [pulsar.IdealPulse(90)],
        observed=pulsar.SpinSite(0.5, 400e6),
        partner=pulsar.SpinSite(0.5, 100.6e6),
        j_hz=128,
        spectral_width_hz=2048,
        points=1024,
        accuracy=2,
    )
    step = actual.x.data[1] - actual.x.data[0]
    for name, axis, amplitude in (
        ("PULSAR (bin centers)", actual.x.data + step / 2, np.abs(actual.data)),
        ("SIMPSON", frequency, reference),
    ):
        peaks = np.argsort(amplitude)[-2:]
        peaks = peaks[np.argsort(axis[peaks])]
        positions = axis[peaks]
        relative = amplitude[peaks] / amplitude[peaks].sum()
        print(f"{name}: peaks {positions} Hz, splitting {np.diff(positions)[0]:.3f} Hz")
        print(f"Relative amplitudes: {relative}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", choices=("mrsimulator", "simpson"))
    args = parser.parse_args()
    if args.reference == "mrsimulator":
        compare_mrsimulator()
    else:
        compare_simpson()
