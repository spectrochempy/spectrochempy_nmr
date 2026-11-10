"""
Explore an alumina spectrum with an extended Czjzek model
=======================================================

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

import matplotlib.pyplot as plt
import numpy as np

from spectrochempy_nmr.czjzek import CzjzekDistribution
from spectrochempy_nmr.czjzek import simulate_czjzek

# %%
# Simulate broad tetrahedral and octahedral environments
# ----------------------------------------------------
# The 12 kHz MAS rate agrees approximately with the archived sideband
# spacing. The stored MASR=5 kHz and CNST31=12.5 kHz differ, so this rate
# should be checked before quantitative interpretation of satellite areas.
larmor_hz = 130.30630648e6
models = (
    (
        "Octahedral model",
        CzjzekDistribution(0.5e6, cq_hz=3e6, eta=0.5, shift_hz=12e-6 * larmor_hz),
    ),
    (
        "Tetrahedral model",
        CzjzekDistribution(0.7e6, cq_hz=4e6, eta=0.5, shift_hz=70e-6 * larmor_hz),
    ),
)

fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
profiles = []
for label, distribution in models:
    spectrum = simulate_czjzek(
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
    frequency_hz = spectrum.x.data
    profiles.append(spectrum.data.real)
    ax.plot(
        frequency_hz / larmor_hz * 1e6,
        profiles[-1] / profiles[-1].max(),
        alpha=0.65,
        label=label,
    )

mixture = 0.75 * profiles[0] + 0.25 * profiles[1]
ax.plot(
    frequency_hz / larmor_hz * 1e6,
    mixture / mixture.max(),
    color="black",
    linewidth=1.5,
    label="Illustrative mixture",
)

# %%
# Optionally inspect the locally supplied experimental central region
# ------------------------------------------------------------------
experiment_path = os.environ.get("SCPY_NMR_ALUMINA_EXPERIMENT")
if experiment_path:
    import nmrglue as ng

    experiment = Path(experiment_path).expanduser()
    metadata, observed = ng.bruker.read_pdata(str(experiment / "pdata" / "1"))
    acqu, proc = metadata["acqus"], metadata["procs"]
    if acqu["NUC1"] != "27Al" or observed.ndim != 1:
        raise ValueError("Expected a processed one-dimensional 27Al experiment.")
    ppm = proc["OFFSET"] - np.arange(observed.size) * (
        proc["SW_p"] / proc["SF"] / observed.size
    )
    center = (ppm > -35) & (ppm < 100)
    if not center.any() or observed[center].max() <= 0:
        raise ValueError("The central 27Al region must have positive signal.")
    ax.plot(
        ppm,
        observed / observed[center].max(),
        color="C3",
        linewidth=1,
        label="Archived alumina, peak normalized",
    )
    ax.set_title("Alumine 800 °C, illustrative model overlay")
else:
    ax.set_title("Illustrative alumina model")
ax.set(
    xlabel="Chemical shift / ppm",
    ylabel="Peak-normalized intensity",
    xlim=(100, -35),
)
ax.legend(fontsize="small")
plt.show()
