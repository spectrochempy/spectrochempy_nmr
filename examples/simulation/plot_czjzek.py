"""
Powder spectra from Czjzek distributions
========================================

The standard Czjzek model describes a statistically isotropic distribution
of electric-field-gradient tensors. Its extended form adds that disorder
to a local tensor. Compare their parameter distributions and the resulting
spin-5/2 central-transition spectra with the same isotropic shift.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

import spectrochempy as scp

# %%
# Define disorder and a persistent local tensor
# ---------------------------------------------
# Frequencies are in Hz. The noise scale is the standard deviation of each
# orthonormal tensor coordinate. A fixed seed keeps these nodes reproducible.
standard = scp.nmr.czjzek.CzjzekDistribution(sigma_hz=0.7e6, shift_hz=5000)
extended = scp.nmr.czjzek.CzjzekDistribution(
    sigma_hz=0.25e6,
    cq_hz=3e6,
    eta=0.6,
    shift_hz=5000,
    shift_sigma_hz=250,
)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
for label, distribution in (("Czjzek", standard), ("Extended Czjzek", extended)):
    nodes = distribution.sample(size=4096, seed=0)
    axes[0].hist(
        np.abs(nodes.cq_hz) / 1e6,
        bins=45,
        density=True,
        histtype="step",
        label=label,
    )
    axes[1].hist(nodes.eta, bins=30, density=True, histtype="step", label=label)
axes[0].set(xlabel=r"$|C_Q|$ / MHz", ylabel="Probability density / MHz⁻¹")
axes[1].set(xlabel=r"$\eta$", ylabel="Probability density")
axes[1].legend()

# %%
# Average the spectra over the distribution
# -----------------------------------------
# This small node count keeps the example quick. Increase both ``samples``
# and powder ``accuracy`` when checking convergence for an actual fit.
fig, ax = plt.subplots(figsize=(8, 3.4), constrained_layout=True)
for label, distribution in (("Czjzek", standard), ("Extended Czjzek", extended)):
    spectrum = scp.nmr.czjzek.simulate_czjzek(
        distribution,
        samples=32,
        seed=0,
        spin=2.5,
        larmor_hz=130e6,
        spinning_hz=12e3,
        spectral_width_hz=160e3,
        points=2049,
        accuracy=4,
        transitions="central",
    )
    ax.plot(spectrum.x.data / 1000, spectrum.data.real, label=label)
ax.set(xlabel="Frequency offset / kHz", ylabel="Bin mass", xlim=(1, 7))
ax.legend()
plt.show()
