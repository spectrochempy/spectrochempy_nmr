"""Qualify cached fast MQMAS kernels on synthetic two-site mixtures.

Includes an exact-model self-consistency control and independent explicit
48-increment States references, with known eta and population fractions.
Use single-threaded BLAS; output includes raw arrays and JSON metrics.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import spectrochempy as scp

from benchmark_pulsar_mqmas import explicit_map
from spectrochempy_nmr.processing.mqmas import _mqmas_geometry


P = scp.nmr.pulsar
INV = scp.nmr.mqmas_inversion
ETA = 0.42
PREPARATION = (P.Pulse(5e-6, 60e3),)
CONVERSION = (P.Pulse(2.5e-6, 70e3), P.Coherence(1))


def dataset(data, *axes):
    result = scp.NDDataset(data)
    coords = dict(x=scp.Coord(axes[-1], units="ppm", rounding=False, sigdigits=15))
    if len(axes) == 2:
        coords["y"] = scp.Coord(axes[0], units="ppm", rounding=False, sigdigits=15)
    result.set_coordset(**coords)
    return result


def summarize(fit, truth, shifts, couplings):
    population = np.asarray(fit.population.data)
    fractions = population / population.sum()
    sites = []
    for delta, pq, fraction in truth:
        region = np.broadcast_to(
            (shifts[:, None] < 1 if delta < 1 else shifts[:, None] >= 1),
            population.shape,
        )
        mass = fractions[region].sum()
        delta_mean = (fractions * shifts[:, None])[region].sum() / mass
        pq_mean = (fractions * couplings[None, :])[region].sum() / mass
        sites.append(
            dict(
                true_shift_ppm=delta,
                true_pq_mhz=pq,
                true_fraction=fraction,
                recovered_shift_ppm=float(delta_mean),
                recovered_pq_mhz=float(pq_mean),
                recovered_fraction=float(mass),
                shift_bias_ppm=float(delta_mean - delta),
                pq_bias_mhz=float(pq_mean - pq),
                fraction_bias=float(mass - fraction),
            )
        )
    return dict(
        rms=float(fit.rms),
        mas_rms=fit.mas_rms,
        mas_gain=fit.mas_gain,
        mas_baseline=fit.mas_baseline,
        sites=sites,
    ), fractions


def full_reference(truth, settings, f1, f2, fwhm1, fwhm2):
    hz_per_ppm = settings["larmor_hz"] * 1e-6
    dx = settings["spectral_width_hz"] / (settings["points"] - 1)
    axis = np.linspace(
        -settings["spectral_width_hz"] / 2,
        settings["spectral_width_hz"] / 2,
        settings["points"],
    )
    lag = np.fft.fftfreq(settings["points"], d=dx)
    gaussian = np.exp(-((np.pi * fwhm2 * lag) ** 2) / (4 * np.log(2)))
    total = np.zeros((f1.size, f2.size))
    for delta, pq, weight in truth:
        full, frequency = explicit_map(
            PREPARATION,
            CONVERSION,
            dict(
                settings,
                shift_hz=delta * hz_per_ppm,
                cq_hz=pq * 1e6 / np.sqrt(1 + ETA**2 / 3),
                eta=ETA,
            ),
            1 / settings["spinning_hz"],
            48,
            f1.size,
            fwhm1,
        )
        np.testing.assert_allclose(
            frequency / hz_per_ppm, f1[::-1], rtol=1e-12, atol=1e-10
        )
        broadened = np.fft.ifft(np.fft.fft(full, axis=1) * gaussian, axis=1)
        # Verify the explicit Fourier broadening against the public operator.
        check = scp.NDDataset(full[0], coordset=[scp.Coord(axis, units="Hz")])
        np.testing.assert_allclose(
            P.broaden(check, gaussian_fwhm_hz=fwhm2).data,
            broadened[0],
            rtol=1e-10,
            atol=1e-16,
        )
        density = broadened.real / (np.diff(frequency).mean() * dx)
        total += weight * np.array(
            [
                np.interp(f2 * hz_per_ppm, axis + dx / 2, row, left=0, right=0)
                for row in density[::-1]
            ]
        )
    return total


def run(output, accuracy=4, full_accuracy=8, workers=2):
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(
        spin=1.5,
        larmor_hz=130.9e6,
        spinning_hz=20000,
        spectral_width_hz=24000,
        points=1024,
        accuracy=accuracy,
        sidebands=0,
        max_rf_step_s=1e-6,
        workers=workers,
    )
    _, _, scale = _mqmas_geometry(1.5, 3)
    f1 = np.fft.fftshift(np.fft.fftfreq(256, 1 / 20000)) / scale / 130.9
    f1 = f1[::-1]
    f2 = np.linspace(18, -18, 128)
    shifts, couplings = np.arange(-4, 7, 2.0), np.linspace(1.2, 2.2, 6)
    fwhm1, fwhm2 = 600, 100
    fast = INV.mqmas_response(
        f1,
        f2,
        PREPARATION,
        CONVERSION,
        f1_gaussian_fwhm_hz=fwhm1,
        f2_gaussian_fwhm_hz=fwhm2,
        **settings,
    )
    blank = dataset(np.zeros((f1.size, f2.size)), f1, f2)
    start = perf_counter()
    kernel = INV.build_kernel(blank, shifts, couplings, fast, asymmetry=ETA)
    rf_build = perf_counter() - start
    print(f"RF library: {rf_build:.3f} s, {fast.cache_info()}", flush=True)
    start = perf_counter()
    INV.build_kernel(
        blank,
        shifts,
        couplings,
        fast.with_broadening(f1_gaussian_fwhm_hz=700),
        asymmetry=ETA,
    )
    cached_build = perf_counter() - start
    assert fast.cache_info().misses == shifts.size * couplings.size
    mas_response = INV.mas_response(
        f2,
        (P.IdealPulse(90),),
        gaussian_fwhm_hz=fwhm2,
        transitions="central",
        **settings,
    )
    start = perf_counter()
    mas_kernel = INV.build_kernel(
        dataset(np.zeros(f2.size), f2), shifts, couplings, mas_response, asymmetry=ETA
    )
    mas_build = perf_counter() - start
    on_grid = [(-2.0, 1.4, 0.65), (4.0, 2.0, 0.35)]
    off_grid = [(-1.5, 1.55, 0.65), (3.5, 1.95, 0.35)]
    weights = np.zeros((shifts.size, couplings.size))
    weights[1, 1] = 0.65
    weights[4, 4] = 0.35
    report = dict(
        accuracy=accuracy,
        full_accuracy=full_accuracy,
        eta=ETA,
        regularization=0,
        f1_gaussian_fwhm_hz=fwhm1,
        f2_gaussian_fwhm_hz=fwhm2,
        rf_library_s=rf_build,
        cached_width_library_s=cached_build,
        mas_library_s=mas_build,
        rf_cache_points=fast.cache_info().currsize,
        matrix_mib=(kernel.matrix.nbytes + mas_kernel.matrix.nbytes) / 1024**2,
        cases={},
    )
    arrays = dict(
        f1_ppm=f1,
        f2_ppm=f2,
        shifts_ppm=shifts,
        pq_mhz=couplings,
        mqmas_kernel=kernel.matrix,
        mas_kernel=mas_kernel.matrix,
    )
    rng = np.random.default_rng(310)
    for name, truth in [
        ("self", on_grid),
        ("full_on_grid", on_grid),
        ("full_off_grid", off_grid),
    ]:
        start = perf_counter()
        if name == "self":
            values = (kernel.matrix @ weights.ravel()).reshape(blank.shape)
            mas_values = mas_kernel.matrix @ weights.ravel()
        else:
            values = full_reference(
                truth, dict(settings, accuracy=full_accuracy), f1, f2, fwhm1, fwhm2
            )
            fine_mas = INV.mas_response(
                f2,
                (P.IdealPulse(90),),
                gaussian_fwhm_hz=fwhm2,
                transitions="central",
                **dict(settings, accuracy=full_accuracy),
            )
            mas_values = sum(
                w * fine_mas(d, pq * 1e6 / np.sqrt(1 + ETA**2 / 3), ETA)
                for d, pq, w in truth
            )
        generation = perf_counter() - start
        if name != "self":
            values += rng.normal(0, 0.001 * np.abs(values).max(), values.shape)
            mas_values += rng.normal(
                0, 0.001 * np.abs(mas_values).max(), mas_values.shape
            )
        observed = dataset(values, f1, f2)
        mas = dataset(1.7 * mas_values + 0.02 * np.abs(mas_values).max(), f2)
        case = dict(
            reference_s=generation,
            noise_fraction=0 if name == "self" else 0.001,
            fits={},
        )
        for kind, options in [
            ("mqmas", {}),
            ("joint", dict(mas=mas, mas_kernel=mas_kernel)),
        ]:
            start = perf_counter()
            fitted = INV.fit(observed, kernel, regularization=0, **options)
            metrics, populations = summarize(fitted, truth, shifts, couplings)
            metrics["fit_s"] = perf_counter() - start
            case["fits"][kind] = metrics
            arrays[name + "_" + kind + "_population"] = populations
            arrays[name + "_" + kind + "_prediction"] = fitted.reconstructed.data
            if fitted.mas_reconstructed is not None:
                arrays[
                    name + "_" + kind + "_mas_prediction"
                ] = fitted.mas_reconstructed.data
        arrays[name + "_observed"] = values
        arrays[name + "_mas_observed"] = mas.data
        report["cases"][name] = case
        print(name, json.dumps(case), flush=True)
    fine_shifts = np.arange(-4, 7, 1.0)
    fine_couplings = np.linspace(1.2, 2.2, 11)
    start = perf_counter()
    fine_kernel = INV.build_kernel(
        blank, fine_shifts, fine_couplings, fast, asymmetry=ETA
    )
    fine_mas_kernel = INV.build_kernel(
        dataset(np.zeros(f2.size), f2),
        fine_shifts,
        fine_couplings,
        mas_response,
        asymmetry=ETA,
    )
    refinement = dict(
        library_s=perf_counter() - start,
        rf_cache_points=fast.cache_info().currsize,
        fits={},
    )
    for kind, options in [
        ("mqmas", {}),
        ("joint", dict(mas=mas, mas_kernel=fine_mas_kernel)),
    ]:
        start = perf_counter()
        fitted = INV.fit(observed, fine_kernel, regularization=0, **options)
        metrics, populations = summarize(fitted, off_grid, fine_shifts, fine_couplings)
        metrics["fit_s"] = perf_counter() - start
        refinement["fits"][kind] = metrics
        arrays["fine_" + kind + "_population"] = populations
        arrays["fine_" + kind + "_prediction"] = fitted.reconstructed.data
        if fitted.mas_reconstructed is not None:
            arrays["fine_" + kind + "_mas_prediction"] = fitted.mas_reconstructed.data
    arrays["fine_shifts_ppm"] = fine_shifts
    arrays["fine_pq_mhz"] = fine_couplings
    arrays["fine_mqmas_kernel"] = fine_kernel.matrix
    arrays["fine_mas_kernel"] = fine_mas_kernel.matrix
    report["refined_off_grid"] = refinement
    print("refinement", json.dumps(refinement), flush=True)
    fast.cache_clear()
    np.savez_compressed(output / "qualification.npz", **arrays)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accuracy", type=int, default=4)
    parser.add_argument("--full-accuracy", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run(args.output, args.accuracy, args.full_accuracy, args.workers)
