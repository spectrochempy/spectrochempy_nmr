"""
Save and reload a Czjzek RF library
===================================

Reuse the same finite-pulse model and compare its interpolated spectrum
with a direct simulation. See the :doc:`guide </userguide/czjzek_fit>`.
"""

# %%
from pathlib import Path
from tempfile import TemporaryDirectory

import spectrochempy as scp

events = (scp.nmr.pulsar.Pulse(0.5e-6, 50e3),)
settings = dict(
    spin=2.5,
    larmor_hz=130e6,
    spectral_width_hz=20e3,
    points=513,
    accuracy=2,
)
grid = dict(
    cq_grid_hz=[i * 250e3 for i in range(-12, 13)],
    eta_grid=[i / 6 for i in range(7)],
    shift_hz=0,
    shift_bins=8,
)
library = scp.nmr.CzjzekRFLibrary.build(events, **grid, **settings)

with TemporaryDirectory() as directory:
    path = Path(directory) / "czjzek-rf.npz"
    library.save(path)
    loaded = scp.nmr.CzjzekRFLibrary.load(path, events, **grid, **settings)

distribution = scp.nmr.czjzek.CzjzekDistribution(150e3, cq_hz=2e6, eta=0.4)
interpolated = loaded.simulate(distribution, samples=16, seed=7)
direct = scp.nmr.czjzek.simulate_czjzek(
    distribution, events=events, samples=16, seed=7, **settings
)
profiles = [
    scp.nmr.pulsar.broaden(s, gaussian_fwhm_hz=400).real for s in (direct, interpolated)
]
_ = scp.plot_multiple(
    profiles,
    labels=["Direct", "Interpolated"],
    method="pen",
    legend="best",
    figsize=(8, 3),
    xlim=(-3000, 1500),
)
