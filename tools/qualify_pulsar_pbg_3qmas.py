"""Fit PBG1 Czjzek ensembles to a resolution- and transfer-weighted 3QMAS map."""

from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path

import nmrglue as ng
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.optimize import nnls

from spectrochempy_nmr.czjzek import CzjzekDistribution
from spectrochempy_nmr.czjzek import simulate_czjzek
from spectrochempy_nmr.pulsar import Pulse
from spectrochempy_nmr.pulsar import _phase_integral
from spectrochempy_nmr.pulsar import _spin_operators
from spectrochempy_nmr.pulsar import _transition_harmonics

from measure_pulsar_pbg_bands import ORDERS
from measure_pulsar_pbg_bands import ROTOR_HZ
from qualify_pulsar_pbg import REGIONS
from qualify_pulsar_pbg import RF_HZ
from qualify_pulsar_pbg import calibrate_center
from qualify_pulsar_pbg import mqmas_parameters
from qualify_pulsar_pbg import ppm_axis
from qualify_pulsar_pbg import ratios
from qualify_pulsar_pbg_fid import read_pbg_fid
from qualify_pulsar_pbg_fid import replay_efp
from qualify_pulsar_pbg_fid import synthesize_fid

SPIN = 2.5
LOCAL_ETA = 0.5
ROTOR_ANGLE = np.arccos(1 / np.sqrt(3))
# Rounded central-only calibration of the earlier 32-node moment model.
MOMENT_WEIGHTS = np.array([0.677, 0.173, 0.035, 0.102, 0.013])
MOMENT_WIDTH_HZ = 1200


def powder_nodes(order):
    """Product quadrature on the rotor-axis hemisphere."""
    z, weights = np.polynomial.legendre.leggauss(order)
    z = (z + 1) / 2
    phi = np.arange(2 * order) * np.pi / order
    radius = np.sqrt(1 - z**2)
    nodes = np.array(
        [
            np.outer(radius, np.cos(phi)),
            np.outer(radius, np.sin(phi)),
            np.broadcast_to(z[:, None], (order, phi.size)),
        ]
    ).reshape(3, -1)
    return nodes, np.repeat(weights / (2 * phi.size), phi.size)


def propagator(offsets, terms, phases, rf_hz, duration_s, max_step_s, ix):
    """Propagate one x pulse through rotor midpoint steps."""
    count = int(np.ceil(duration_s / max_step_s))
    step = duration_s / count
    result = np.broadcast_to(np.eye(6), (*offsets.shape[:-1], phases.size, 6, 6))
    diagonal = np.arange(6)
    orders = np.arange(1, 5)
    for index in range(count):
        phase = phases + 2 * np.pi * ROTOR_HZ * step * (index + 0.5)
        rotation = np.exp(-1j * orders[:, None] * phase)
        differences = offsets[:, None, :] + (terms @ rotation).real.swapaxes(-1, -2)
        hamiltonian = np.broadcast_to(rf_hz * ix, result.shape).copy()
        hamiltonian[..., diagonal[1:], diagonal[1:]] = differences.cumsum(axis=-1)
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        factor = np.exp(-2j * np.pi * step * eigenvalues)
        evolution = (eigenvectors * factor[..., None, :]) @ eigenvectors.swapaxes(
            -1, -2
        )
        result = evolution @ result
    return result, phases + 2 * np.pi * ROTOR_HZ * duration_s


def transfer(
    cq_hz, eta, directions, phases, larmor_hz, durations, rf, max_step, detuning=0
):
    """Return CT frequencies and complex inner-3Q-to-CT amplitudes per orientation."""
    harmonics = np.stack(
        [
            terms.T
            for _, _, terms in _transition_harmonics(
                directions, cq_hz, eta, ROTOR_ANGLE, SPIN, larmor_hz, 0.0
            )
        ],
        axis=1,
    )
    offsets, terms = harmonics[..., 0].real + detuning, harmonics[..., 1:]
    ix, _, iz = _spin_operators(SPIN)
    p1, p2, p3, d0, zfilter = durations
    excite, phase1 = propagator(offsets, terms, phases, rf[0], p1, max_step, ix)
    rho = excite @ iz @ excite.conj().swapaxes(-1, -2)
    phase2 = phase1 + 2 * np.pi * ROTOR_HZ * d0
    t1_phase = 2 * np.pi * offsets[:, None, 1:4].sum(axis=-1) * d0 + (
        _phase_integral(terms[:, 1:4], phase2, ROTOR_HZ)
        - _phase_integral(terms[:, 1:4], phase1, ROTOR_HZ)
    ).sum(axis=-1)
    # With descending m, rho[4, 1] acquires the negative phase from
    # level[4]-level[1]. The conjugate pathway completes the populations.
    inner = rho[..., 4, 1] * np.exp(-1j * t1_phase)
    convert, phase3 = propagator(offsets, terms, phase2, rf[0], p2, max_step, ix)
    populations = (
        2 * (convert[..., :, 4] * inner[..., None] * convert[..., :, 1].conj()).real
    )
    phase3 = phase3 + 2 * np.pi * ROTOR_HZ * zfilter
    read, phase4 = propagator(offsets, terms, phase3, rf[1], p3, max_step, ix)
    ct = (
        -1j
        * 3
        * np.sum(read[..., 2, :] * populations * read[..., 3, :].conj(), axis=-1)
    )
    # Order-zero direct MAS response, including the rotor phase at readout.
    ct_terms = terms[:, 2:3]
    wave = np.exp(1j * _phase_integral(ct_terms, phase4, ROTOR_HZ)[..., 0])
    ct = (ct * wave.conj()).mean(axis=-1) * wave.mean(axis=-1)
    return harmonics[:, 2, 0].real, ct


def transfer_table(metadata, larmor_hz, settings):
    acqu = metadata["acqus"]
    p1, p2, p3 = np.asarray(acqu["P"][1:4]) * 1e-6
    zfilter = acqu["L"][30] / acqu["CNST"][31] - (p2 + p3) / 2
    # Use the stored pulse durations; the title supplies nominal RF at these powers.
    rf = settings.rf_scale * np.array([100_000, 10_000])
    directions, powder_weights = powder_nodes(settings.powder_order)
    phases = np.arange(settings.rotor_count) * (2 * np.pi / settings.rotor_count)
    cq_grid = np.arange(int(np.ceil(20 / settings.cq_step)) + 1) * settings.cq_step
    eta_grid = np.linspace(0, 1, 9)
    detunings = np.linspace(-20_000.0, 20_000.0, 40 // settings.detuning_step_khz + 1)
    offsets = np.empty((cq_grid.size, eta_grid.size, powder_weights.size))
    values = np.empty(
        (cq_grid.size, eta_grid.size, detunings.size, powder_weights.size)
    )
    for i, cq in enumerate(cq_grid):
        for j, eta in enumerate(eta_grid):
            if settings.uniform_transfer:
                ct = next(
                    terms[0].real
                    for m, _, terms in _transition_harmonics(
                        directions, cq * 1e6, eta, ROTOR_ANGLE, SPIN, larmor_hz, 0.0
                    )
                    if m == 0.5
                )
            else:
                # Reversing the quadrupolar sign mirrors the single-quantum
                # offsets in spin order, leaving this real-x pathway invariant.
                for k, detuning in enumerate(detunings):
                    ct, weight = transfer(
                        cq * 1e6,
                        eta,
                        directions,
                        phases,
                        larmor_hz,
                        (p1, p2, p3, 1e-6, zfilter),
                        rf,
                        settings.max_step_us * 1e-6,
                        detuning,
                    )
                    values[i, j, k] = weight.real * powder_weights
            offsets[i, j] = ct * 1e6 / larmor_hz
    offsets = RegularGridInterpolator((cq_grid, eta_grid), offsets)
    if settings.uniform_transfer:
        return offsets, None, powder_weights
    # Set the global absorptive sign from the simulated low-coupling response.
    sign = np.sign(
        values[1 : round(5 / settings.cq_step) + 1, :, detunings.size // 2].sum()
    )
    if not sign:
        raise ValueError("No observable 3Q transfer in the lookup table.")
    return (
        offsets,
        RegularGridInterpolator((cq_grid, eta_grid, detunings), sign * values),
        powder_weights,
    )


def f1_kernel(metadata, offsets_ppm):
    """Signed absorptive F1 response of the finite States acquisition."""
    proc, acqu = metadata["proc2s"], metadata["acqu2s"]
    dwell = 1 / acqu["SW_h"]
    time = np.arange(min(proc["TDeff"], acqu["TD"]) // 2) * dwell
    window = np.exp(-np.pi * proc["LB"] * time)
    window[0] *= proc["FCOR"]
    phase = 2 * np.pi * proc["SF"] * offsets_ppm[..., None] * time
    return (np.cos(phase) @ window) / window.sum()


def distribution(parameters, larmor_hz, carrier_hz=0.0):
    shift, pq_mhz, local_fraction, shift_sigma = parameters
    pq_hz = pq_mhz * 1e6
    return CzjzekDistribution(
        sigma_hz=pq_hz * np.sqrt(0.3 * (1 - local_fraction)),
        cq_hz=pq_hz * np.sqrt(local_fraction / (1 + LOCAL_ETA**2 / 3)),
        eta=LOCAL_ETA,
        shift_hz=shift * larmor_hz * 1e-6 - carrier_hz,
        shift_sigma_hz=shift_sigma * larmor_hz * 1e-6,
    )


def direct_profiles(grid, centers, weights, sigma_ppm):
    """Deposit each tensor's orientation masses on the F2 grid and broaden."""
    step = grid[1] - grid[0]
    position = (centers - grid[0]) / step
    lower = np.floor(position).astype(int)
    fraction = position - lower
    result = np.zeros((centers.shape[0], grid.size))
    rows = np.broadcast_to(np.arange(centers.shape[0])[:, None], centers.shape)
    for bins, mass in (
        (lower, (1 - fraction) * weights),
        (lower + 1, fraction * weights),
    ):
        keep = (bins >= 0) & (bins < grid.size)
        np.add.at(result, (rows[keep], bins[keep]), mass[keep])
    return gaussian_filter1d(result, sigma_ppm / step, axis=1, mode="constant")


def fit_map(experiment, settings):
    metadata, signal = ng.bruker.read_pdata(str(experiment / "pdata" / "1"))
    p1, p2 = metadata["proc2s"], metadata["procs"]
    if (
        signal.shape != (128, 1024)
        or metadata["acqu2s"]["FnMODE"] != 4
        or p1["WDW"] != 1
        or not np.isclose(p1["SF"] / p2["SF"], 17 / 12)
        or metadata["acqus"]["PULPROG"] != "pf_3Qzf"
    ):
        raise ValueError("Expected the archived PBG1 States 3QMAS map.")
    f1, f2 = ppm_axis(p1)[::-1], ppm_axis(p2)[::-1]
    signal = signal[::-1, ::-1]
    # Use signed intensities and a row-wise baseline outside all site boxes.
    signal = signal - np.median(signal[:, (f2 > -65) & (f2 < -40)], axis=1)[:, None]
    mask = np.zeros_like(signal, dtype=bool)
    for xmin, xmax, ymin, ymax in REGIONS:
        mask |= ((f1 > xmin) & (f1 < xmax))[:, None] & ((f2 > ymin) & (f2 < ymax))
    target = signal[mask]
    target /= target.max()
    larmor = p2["SF"] * 1e6
    carrier = metadata["acqus"]["SFO1"] * 1e6 - larmor
    ct_offsets, efficiency, powder_weights = transfer_table(metadata, larmor, settings)
    print(
        f"Computed orientation table, uniform transfer={settings.uniform_transfer}",
        flush=True,
    )
    centers = mqmas_parameters(experiment)

    @lru_cache(maxsize=32)
    def component(region, shift, pq, local_amplitude, shift_sigma):
        d = distribution((shift, pq, local_amplitude**2, shift_sigma), larmor)
        member = d.sample(settings.fit_samples, settings.seed + region)
        coords = np.column_stack((np.abs(member.cq_hz) / 1e6, member.eta))
        delta = member.shift_hz * 1e6 / larmor
        y = 6000 * (member.cq_hz / larmor) ** 2 * (1 + member.eta**2 / 3)
        indirect = f1_kernel(metadata, f1[:, None] - (delta + 10 / 17 * y))
        direct_centers = delta[:, None] + ct_offsets(coords)
        weights = (
            efficiency(np.column_stack((coords, member.shift_hz - carrier)))
            if efficiency is not None
            else powder_weights
        )
        direct = direct_profiles(f2, direct_centers, weights, settings.f2_sigma_ppm)
        return ((indirect * member.weights) @ direct)[mask]

    def design(parameters):
        return np.column_stack(
            [component(i, *row) for i, row in enumerate(parameters.reshape(-1, 4))]
        )

    def residual(parameters):
        profiles = design(parameters)
        weights, _ = nnls(profiles, target)
        return profiles @ weights - target

    low, high, initial = [], [], []
    for shift, pq in centers:
        low.extend((shift - 4, 0.5, 0.0, 0.05))
        high.extend((shift + 4, 7.0, 1.0, 8.0))
        initial.extend((shift, pq / 1e6, np.sqrt(0.5), 1.5))
    bounds = (low, high)
    result = least_squares(residual, initial, bounds=bounds, max_nfev=settings.max_nfev)
    print(
        f"3QMAS normalized RMS={np.sqrt(np.mean(result.fun**2)):.5f}, "
        f"nfev={result.nfev}, status={result.status}",
        flush=True,
    )
    parameters = result.x.reshape(-1, 4).copy()
    parameters[:, 2] **= 2
    amplitudes, _ = nnls(design(result.x), target)
    print(f"3QMAS amplitudes={np.round(amplitudes, 5)}", flush=True)
    print(f"3QMAS active bounds={result.active_mask.reshape(-1, 4)}", flush=True)
    return parameters


def compare(pbg1, parameters, samples, seed):
    acqu, proc, frequency, published, fid = read_pbg_fid(pbg1 / "1")
    larmor_hz = proc["SF"] * 1e6
    carrier_hz = (acqu["SFO1"] - proc["SF"]) * 1e6
    pulse = Pulse(acqu["P"][1] * 1e-6, RF_HZ)
    profiles = []
    for index, row in enumerate(parameters):
        print(
            f"region {index + 1}: shift, Pq RMS, local fraction, shift sigma = {row}",
            flush=True,
        )
        profile = simulate_czjzek(
            distribution(row, larmor_hz, carrier_hz),
            spin=SPIN,
            larmor_hz=larmor_hz,
            spectral_width_hz=4e6,
            samples=samples,
            seed=seed + index,
            points=65537,
            accuracy=4,
            spinning_hz=ROTOR_HZ,
            sidebands=20,
            transitions="all",
            events=[pulse],
            max_rf_step_s=0.125e-6,
        )
        profiles.append(profile.data)
    axis = profile.x.data
    axis = axis + (axis[1] - axis[0]) / 2 + carrier_hz
    components = np.column_stack(profiles)
    rms, width, weights, _ = calibrate_center(
        frequency, published, axis, components.real
    )

    def replay(profile, width_hz):
        modeled = synthesize_fid(axis, profile, width_hz, acqu, proc, acqu["DE"] * 1e-6)
        return ratios(frequency, replay_efp(modeled, acqu, proc).real)

    measured = ratios(frequency, replay_efp(fid, acqu, proc).real)
    predicted = replay(components @ weights, width)
    fixed = replay(components @ MOMENT_WEIGHTS, MOMENT_WIDTH_HZ)
    satellite = np.asarray(ORDERS) != 0
    errors = abs(measured[satellite] - predicted[satellite])
    print(
        f"central RMS={rms:.2%}, FWHM={width}, weights={weights}; "
        f"sums={measured[satellite].sum():.3f}/{predicted[satellite].sum():.3f}%; "
        f"area MAE/max={errors.mean():.3f}/{errors.max():.3f} pp",
        flush=True,
    )
    fixed_errors = abs(measured[satellite] - fixed[satellite])
    print(
        f"fixed earlier weights/width: sum={fixed[satellite].sum():.3f}%, "
        f"area MAE/max={fixed_errors.mean():.3f}/{fixed_errors.max():.3f} pp",
        flush=True,
    )
    print("order,measured_pct,czjzek_pct,fixed_weights_pct")
    for order, observed, calculated, fixed_area in zip(
        ORDERS, measured, predicted, fixed, strict=True
    ):
        print(f"{order:+d},{observed:.3f},{calculated:.3f},{fixed_area:.3f}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("pbg1", type=Path)
    parser.add_argument("--fit-samples", type=int, default=64)
    parser.add_argument(
        "--samples", type=int, default=16, help="1D Sobol nodes per region"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--powder-order", type=int, default=5)
    parser.add_argument("--rotor-count", type=int, default=12)
    parser.add_argument("--cq-step", type=float, default=0.5)
    parser.add_argument("--detuning-step-khz", type=int, choices=(5, 10), default=10)
    parser.add_argument("--rf-scale", type=float, default=1.0)
    parser.add_argument("--max-step-us", type=float, default=0.25)
    parser.add_argument("--f2-sigma-ppm", type=float, default=0.6)
    parser.add_argument("--uniform-transfer", action="store_true")
    parser.add_argument("--max-nfev", type=int, default=60)
    fit_files = parser.add_mutually_exclusive_group()
    fit_files.add_argument("--fit-input", type=Path, help="Read saved 3QMAS parameters")
    fit_files.add_argument(
        "--fit-output", type=Path, help="Save fitted 3QMAS parameters"
    )
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    if args.fit_input:
        params = np.loadtxt(args.fit_input)
        if params.shape != (len(REGIONS), 4):
            parser.error("Expected five rows of four fitted parameters.")
    else:
        params = fit_map(args.pbg1 / "13", args)
        if args.fit_output:
            np.savetxt(args.fit_output, params)
    print(params, flush=True)
    if not args.fit_only:
        compare(args.pbg1, params, args.samples, args.seed)


if __name__ == "__main__":
    main()
