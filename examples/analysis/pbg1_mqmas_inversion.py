"""Compare PBG1 MQMAS and joint MQMAS/MAS inversion
================================================

Run this example on experiments 13 and 1 from Zenodo record 17130697.
The accompanying guide explains the acquisition and the model assumptions.
"""

from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import spectrochempy as scp


REGIONS = (
    (-14, 0, -22, 3),
    (1, 14, -8, 13),
    (37, 46, 32, 46),
    (44, 58, 41, 57),
    (63, 77, 62, 77),
)


def read_spectra(directory):
    """Select the signed RR map and the processed MAS central window."""
    processed = scp.nmr.read_topspin(directory / "13/pdata/1/2rr", proc_axis=True)
    corrected = scp.nmr.Experiment(processed).abs2(order=0, ranges=[(-65, -40)])
    rr = scp.nmr.Experiment(corrected).rr()
    mqmas = rr[float(rr.y.data.max()) : -16.0, 85.0:-26.0][:, ::2]
    selected = np.zeros(mqmas.shape, dtype=bool)
    for low_y, high_y, low_x, high_x in REGIONS:
        selected |= ((mqmas.y.data > low_y) & (mqmas.y.data < high_y))[:, None] & (
            (mqmas.x.data > low_x) & (mqmas.x.data < high_x)
        )[None, :]
    mqmas.mask = ~selected
    mas = scp.nmr.read_topspin(directory / "1/pdata/1/1r", proc_axis=True).real
    limit = 14000 / mas.meta.sf[-1].to("MHz").magnitude
    mas = mas[float(limit) : -float(limit)]
    return mqmas, mas


def build_kernels(mqmas, mas, shifts, products, eta_nodes):
    """Construct per-population responses with the public inversion API."""
    inversion = scp.nmr.mqmas_inversion
    pulse = scp.nmr.pulsar.Pulse
    metadata = mqmas.meta.source_metadata
    acqu, indirect = metadata["acqus"], metadata["acqu2s"]
    proc, proc1 = metadata["procs"], metadata["proc2s"]
    reference = proc["SF"] * 1e6
    carrier = (acqu["SFO1"] * 1e6 - reference) / (reference * 1e-6)
    p1, p2, p3 = np.asarray(acqu["P"][1:4]) * 1e-6
    response_3q = inversion.z_filtered_3qmas_response(
        mqmas.y.data,
        mqmas.x.data,
        [pulse(p1, 100000), pulse(p2, 100000), pulse(p3, 10000)],
        spin=2.5,
        larmor_hz=reference,
        spinning_hz=acqu["MASR"],
        carrier_ppm=carrier,
        conversion_delay_s=1e-6,
        z_filter_s=acqu["L"][30] / acqu["CNST"][31] - (p2 + p3) / 2,
        dwell_s=1 / indirect["SW_h"],
        increments=min(indirect["TD"], proc1["TDeff"]) // 2,
        f1_lb_hz=proc1["LB"],
        first_point_scale=proc1["FCOR"],
        f2_sigma_ppm=0.6,
        receiver_phase_deg=180,
        powder_order=5,
        rotor_count=12,
        max_rf_step_s=0.25e-6,
    )
    print("Building the 3QMAS kernel", flush=True)
    kernel_3q = inversion.build_kernel(
        mqmas, shifts, products, response_3q, asymmetry="uniform", nodes=eta_nodes
    )
    metadata = mas.meta.source_metadata
    acqu, proc = metadata["acqus"], metadata["procs"]
    reference = proc["SF"] * 1e6
    carrier = (acqu["SFO1"] * 1e6 - reference) / (reference * 1e-6)
    response_mas = inversion.mas_response(
        mas.x.data,
        [pulse(acqu["P"][1] * 1e-6, 50000)],
        spin=2.5,
        larmor_hz=reference,
        spinning_hz=acqu["MASR"],
        carrier_ppm=carrier,
        spectral_width_hz=500000,
        points=8193,
        accuracy=5,
        rf_solver="tridiagonal",
        sidebands=0,
        rotor_angle_deg=np.rad2deg(np.arccos(1 / np.sqrt(3))),
        gaussian_fwhm_hz=1600,
        max_rf_step_s=0.125e-6,
    )
    print("Building the MAS kernel", flush=True)
    kernel_mas = inversion.build_kernel(
        mas, shifts, products, response_mas, asymmetry="uniform", nodes=eta_nodes
    )
    return kernel_3q, kernel_mas


def compare(mqmas, mas, kernel_3q, kernel_mas):
    """Use one grid and one regularization for the two inversions."""
    inversion = scp.nmr.mqmas_inversion
    mq_only = inversion.fit(mqmas, kernel_3q, regularization=0.1)
    joint = inversion.fit(
        mqmas,
        kernel_3q,
        regularization=0.1,
        mas=mas,
        mas_kernel=kernel_mas,
        mas_weight=1.0,
    )
    prediction = mas.copy()
    prediction.data = kernel_mas.matrix @ mq_only.population.data.ravel()
    centered = prediction - prediction.mean()
    gain = float(((mas - mas.mean()) * centered).sum() / (centered**2).sum())
    if gain <= 0:
        raise ValueError("The 3QMAS-only population requires a nonpositive MAS gain.")
    prediction = gain * centered + mas.mean()
    return mq_only, joint, prediction


def summarize(mq_only, joint, mas, prediction):
    """Report errors and integrated model weights in the negative-shift region."""
    result = {}
    for label, fit, predicted in (
        ("3QMAS only", mq_only, prediction),
        ("Joint", joint, joint.mas_reconstructed),
    ):
        population = fit.population
        region = (population.y.data >= -14) & (population.y.data <= 0)
        low = population.x.data < 1.5
        local = float(population.data[np.ix_(region, low)].sum())
        total = float(population.sum())
        residual = (mas - predicted) / abs(mas).max()
        result[label] = dict(
            mqmas_rms=float(fit.rms),
            mas_rms=float((residual**2).mean() ** 0.5),
            local_weight=local,
            local_fraction=local / total,
            total_weight=total,
        )
    return result


def plot_comparison(directory, mqmas, mas, mq_only, joint, prediction):
    """Export comparisons through SpectroChemPy's plotting API."""
    populations = [mq_only.population.T, joint.population.T]
    high = max(float(p.max()) for p in populations)
    _ = scp.multiplot(
        populations,
        ncol=2,
        method="image",
        labels=["3QMAS only", "Joint 3QMAS + MAS"],
        cmap="viridis",
        zlabel="Model weight per cell",
        vmin=0,
        vmax=high,
        colorbar=True,
        equal_aspect=False,
        figsize=(10, 4),
        show=False,
        dpi=150,
        output=directory / "populations.png",
    )
    peak = abs(mqmas).max()
    _ = scp.multiplot(
        [mqmas / peak, mq_only.residual / peak, joint.residual / peak],
        ncol=3,
        method="image",
        labels=["Observed 3QMAS", "3QMAS-only residual", "Joint residual"],
        cmap="RdBu_r",
        zlabel="Fraction of 3QMAS peak",
        panel_kwargs=[
            dict(vmin=-1, vmax=1),
            dict(vmin=-0.2, vmax=0.2),
            dict(vmin=-0.2, vmax=0.2),
        ],
        colorbar=True,
        equal_aspect=False,
        figsize=(12, 4),
        show=False,
        dpi=150,
        output=directory / "mqmas-residuals.png",
    )
    _ = scp.plot_multiple(
        [mas, prediction, joint.mas_reconstructed],
        method="pen",
        labels=["Observed MAS", "From 3QMAS-only population", "Joint"],
        legend="best",
        ylabel="MAS intensity (a.u.)",
        figsize=(9, 4),
        show=False,
        dpi=150,
        output=directory / "mas-comparison.png",
    )


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("pbg1", type=Path, help="Extracted PBG1_2,5mm directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shift-points", type=int, default=49)
    parser.add_argument("--pq-points", type=int, default=11)
    parser.add_argument("--eta-nodes", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    mqmas, mas = read_spectra(args.pbg1)
    shifts = scp.Coord.linspace(-16, 80, args.shift_points, units="ppm")
    products = scp.Coord.linspace(0.5, 8, args.pq_points, units="MHz")
    kernels = build_kernels(mqmas, mas, shifts.data, products.data, args.eta_nodes)
    mq_only, joint, prediction = compare(mqmas, mas, *kernels)
    metrics = summarize(mq_only, joint, mas, prediction)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        args.output / "comparison.npz",
        shift_ppm=shifts.data,
        pq_mhz=products.data,
        population_mqmas=mq_only.population.data,
        population_joint=joint.population.data,
        f1_ppm=mqmas.y.data,
        f2_ppm=mqmas.x.data,
        fit_mask=~mqmas.mask,
        observed_mqmas=mqmas.data,
        predicted_mqmas=mq_only.reconstructed.data,
        predicted_joint=joint.reconstructed.data,
        mas_ppm=mas.x.data,
        observed_mas=mas.data,
        predicted_mas_from_mqmas=prediction.data,
        predicted_mas_joint=joint.mas_reconstructed.data,
    )
    plot_comparison(args.output, mqmas, mas, mq_only, joint, prediction)
    print(json.dumps(metrics, indent=2))
    print(f"Elapsed time {perf_counter() - start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
