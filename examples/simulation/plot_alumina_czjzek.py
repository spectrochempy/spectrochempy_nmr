"""
Explore an alumina spectrum with an extended Czjzek model
=========================================================

An archived Bruker acquisition in Dropbox Storage is titled
``Alumine 800 C`` and records 27Al at 130.306 MHz. Its central region is a
candidate for exploring disorder models. This example displays a model
with two aluminum environments and optionally overlays that processed
spectrum supplied locally.

Set ``SCPY_NMR_ALUMINA_EXPERIMENT`` to the experiment directory containing
``acqus`` and ``pdata/1`` to enable the experimental overlay. Without it,
the example displays the simulated components. The model parameters below
are illustrative and the plot uses peak-normalized intensities.
"""

# %%
import os
from pathlib import Path

import spectrochempy as scp

# %%
# Simulate broad tetrahedral and octahedral environments
# ------------------------------------------------------
# The 12 kHz MAS rate agrees approximately with the archived sideband
# spacing. The stored MASR=5 kHz and CNST31=12.5 kHz differ, so this rate
# should be checked before quantitative interpretation of satellite areas.
larmor_hz = 130.30630648e6
models = (
    (
        "Octahedral model",
        scp.nmr.czjzek.CzjzekDistribution(
            0.5e6, cq_hz=3e6, eta=0.5, shift_hz=12e-6 * larmor_hz
        ),
    ),
    (
        "Tetrahedral model",
        scp.nmr.czjzek.CzjzekDistribution(
            0.7e6, cq_hz=4e6, eta=0.5, shift_hz=70e-6 * larmor_hz
        ),
    ),
)

profiles = []
labels = []
for label, distribution in models:
    spectrum = scp.nmr.czjzek.simulate_czjzek(
        distribution,
        samples=32,
        seed=0,
        spin=2.5,
        larmor_hz=larmor_hz,
        spinning_hz=12e3,
        spectral_width_hz=200e3,
        points=4097,
        accuracy=4,
        transitions="central",
    )
    spectrum.x.meta["acquisition_frequency"] = larmor_hz * scp.ur.Hz
    spectrum.x.ito("ppm")
    profiles.append(spectrum.real)
    labels.append(label)

mixture = 0.75 * profiles[0] + 0.25 * profiles[1]
spectra = [profile.normalize() for profile in (*profiles, mixture)]
labels.append("Illustrative mixture")
colors = ["C0", "C1", "black"]
widths = [1.5, 1.5, 1.5]
alphas = [0.65, 0.65, 1.0]

# %%
# Optionally inspect the locally supplied experimental central region
# -------------------------------------------------------------------
experiment_path = os.environ.get("SCPY_NMR_ALUMINA_EXPERIMENT")
title = "Illustrative alumina model"
if experiment_path:
    experiment = Path(experiment_path).expanduser()
    observed = scp.nmr.read_topspin(experiment / "pdata" / "1" / "1r", proc_axis=True)
    if observed is None:
        raise ValueError("Expected a processed one-dimensional 27Al experiment.")
    observed = observed.squeeze().real
    if observed.ndim != 1 or observed.meta.nuc1[-1] != "27Al":
        raise ValueError("Expected a processed one-dimensional 27Al experiment.")
    ppm = observed.x.data
    center = (ppm > -35) & (ppm < 100)
    if not center.any():
        raise ValueError("The central 27Al region must have positive signal.")
    peak = observed[center].max()
    if peak.magnitude <= 0:
        raise ValueError("The central 27Al region must have positive signal.")
    spectra.append(observed / peak)
    labels.append("Archived alumina, peak normalized")
    colors.append("C3")
    widths.append(1)
    alphas.append(1.0)
    title = "Alumine 800 °C, illustrative model overlay"

_ = scp.plot_multiple(
    spectra,
    method="pen",
    labels=labels,
    color=colors,
    lw=widths,
    alpha=alphas,
    legend="best",
    figsize=(8, 3.8),
    xlabel="Chemical shift / ppm",
    ylabel="Peak-normalized intensity",
    xlim=(-35, 100),
    reverse=True,
    title=title,
)
