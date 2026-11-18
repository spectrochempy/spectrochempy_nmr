"""Compare direct MQMAS maps with explicit symmetric-coherence States series.

Use OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1.
Both models use the same spin system, pulses, RF step, powder mesh and
F2 grid. Full series are sheared with unit chemical-shift sensitivity;
Gaussian F1 apodization matches the direct model's F1 Gaussian.
"""

import argparse
import json
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from statistics import median
from time import perf_counter, process_time

import numpy as np
from spectrochempy.nmr import pulsar
from spectrochempy_nmr.processing.mqmas import _mqmas_geometry


CASES = {
    "al-sync": dict(
        spin=2.5,
        cq_hz=2.923e6,
        eta=0.136,
        shift_hz=1000,
        spinning_hz=12500,
        larmor_hz=104.46851646e6,
        excitation=(12e-6, 110e3),
        conversion=(4e-6, 85e3),
    ),
    "al-unsync": dict(
        spin=2.5,
        cq_hz=2.923e6,
        eta=0.136,
        shift_hz=1000,
        spinning_hz=12500,
        larmor_hz=104.46851646e6,
        excitation=(12e-6, 110e3),
        conversion=(4e-6, 85e3),
        dwell=7.5e-6,
        f1_fwhm_hz=6000,
    ),
    "rb-sync": dict(
        spin=1.5,
        cq_hz=1.8e6,
        eta=0.42,
        shift_hz=300,
        spinning_hz=20000,
        larmor_hz=130.9e6,
        excitation=(5e-6, 60e3),
        conversion=(2.5e-6, 70e3),
    ),
}


def explicit_map(preparation, conversion, settings, dwell, increments, f1_points, fwhm):
    """Reconstruct F1 separately for the real and imaginary F2 channels."""
    _, ratio, scale = _mqmas_geometry(settings["spin"], 3)
    times = np.arange(increments) * dwell
    paths = tuple(
        (*preparation, pulsar.Coherence(sign * 3, symmetric=True)) for sign in (1, -1)
    )
    states = [
        pulsar.simulate_t1_series(
            pulsar.PhaseCycle(paths, weights),
            times,
            conversion,
            **settings,
            transitions="central",
            rotor_angle_deg=np.rad2deg(np.arccos(1 / np.sqrt(3))),
        )
        for weights in ((0.5, 0.5), (-0.5j, 0.5j))
    ]
    cosine, sine = [state.data for state in states]
    sigma = fwhm / np.sqrt(8 * np.log(2))
    window = np.exp(-2 * np.pi**2 * (sigma * scale * times) ** 2)
    window[0] *= 0.5
    shear = np.exp(-2j * np.pi * ratio * times[:, None] * states[0].x.data)
    channels = (cosine.real + 1j * sine.real, cosine.imag + 1j * sine.imag)
    transformed = [
        np.fft.fftshift(
            np.fft.fft(channel * shear * window[:, None], n=f1_points, axis=0), axes=0
        )
        for channel in channels
    ]
    data = 2 / f1_points * (transformed[0].real + 1j * transformed[1].real)
    return data, np.fft.fftshift(np.fft.fftfreq(f1_points, dwell)) / scale


def run(case, accuracy, full_accuracy, workers, output, increments=48, f1_points=512):
    specification = dict(CASES[case])
    prep = (pulsar.Pulse(*specification.pop("excitation")),)
    conversion = (pulsar.Pulse(*specification.pop("conversion")), pulsar.Coherence(1))
    dwell = specification.pop("dwell", 1 / specification["spinning_hz"])
    fwhm = specification.pop("f1_fwhm_hz", 600)
    settings = dict(
        specification,
        spectral_width_hz=24000,
        points=2048,
        accuracy=accuracy,
        sidebands=0,
        max_rf_step_s=1e-6,
        workers=workers,
        deposition="numba",
    )
    _, _, scale = _mqmas_geometry(settings["spin"], 3)
    spectrum_settings = dict(
        f1_spectral_width_hz=1 / (dwell * scale),
        f1_points=f1_points,
        f1_gaussian_fwhm_hz=fwhm,
    )
    # Initialize optional backend and powder machinery outside the timed calls.
    pulsar.MQMASModel(prep, conversion, **dict(settings, accuracy=1))
    start, cpu = perf_counter(), process_time()
    model = pulsar.MQMASModel(prep, conversion, **settings)
    build_time, build_cpu = perf_counter() - start, process_time() - cpu
    render_times = []
    for _ in range(5):
        start = perf_counter()
        fast = model.spectrum(**spectrum_settings)
        render_times.append(perf_counter() - start)
    print(
        f"{case}: RF build {build_time:.3f} s; rendering {median(render_times):.6f} s",
        flush=True,
    )
    start, cpu = perf_counter(), process_time()
    full, f1 = explicit_map(
        prep,
        conversion,
        dict(settings, accuracy=full_accuracy),
        dwell,
        increments,
        f1_points,
        fwhm,
    )
    full_time, full_cpu = perf_counter() - start, process_time() - cpu
    np.testing.assert_allclose(fast.y.data, f1, rtol=1e-12, atol=1e-10)
    delta = fast.data - full
    report = dict(
        case=case,
        accuracy=accuracy,
        full_accuracy=full_accuracy,
        increments=increments,
        dwell_s=dwell,
        f1_gaussian_fwhm_hz=fwhm,
        final_window_weight=float(
            np.exp(
                -2
                * np.pi**2
                * (fwhm / np.sqrt(8 * np.log(2)) * scale * (increments - 1) * dwell)
                ** 2
            )
        ),
        workers=workers,
        f1_position_hz=model.f1_position_hz,
        rf_build_s=build_time,
        rf_build_cpu_s=build_cpu,
        render_median_s=median(render_times),
        full_s=full_time,
        full_cpu_s=full_cpu,
        process_peak_rss_mib=getrusage(RUSAGE_SELF).ru_maxrss / 1024,
        relative_l2=float(np.linalg.norm(delta) / np.linalg.norm(full)),
        relative_peak_error=float(np.abs(delta).max() / np.abs(full).max()),
        relative_f2_projection_l2=float(
            np.linalg.norm(delta.sum(axis=0)) / np.linalg.norm(full.sum(axis=0))
        ),
        fast_area=[float(fast.data.sum().real), float(fast.data.sum().imag)],
        full_area=[float(full.sum().real), float(full.sum().imag)],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        output.with_suffix(".npz"),
        fast=fast.data,
        full=full,
        f1=fast.y.data,
        f2=fast.x.data,
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--accuracy", type=int, default=4)
    parser.add_argument("--full-accuracy", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.case,
        args.accuracy,
        args.full_accuracy or args.accuracy,
        args.workers,
        args.output,
    )
