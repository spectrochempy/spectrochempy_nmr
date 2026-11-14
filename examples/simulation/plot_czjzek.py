"""
Powder spectra from Czjzek distributions
========================================

The standard Czjzek model describes a statistically isotropic distribution
of electric-field-gradient tensors. Its extended form adds that disorder
to a local tensor. Compare their parameter distributions and the resulting
spin-5/2 central-transition spectra with the same isotropic shift.
"""

# %%
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

labels = ["Czjzek", "Extended Czjzek"]
couplings = []
asymmetries = []
for distribution in (standard, extended):
    nodes = distribution.sample(size=4096, seed=0)
    couplings.append(
        scp.NDDataset(abs(nodes.cq_hz), units="Hz", title=r"$|C_Q|$").to("MHz")
    )
    asymmetries.append(scp.NDDataset(nodes.eta, title=r"$\eta$"))

# Overlay each pair of probability densities on a separate set of axes.
for samples, bins in ((couplings, 45), (asymmetries, 30)):
    ax = None
    for sample, label in zip(samples, labels, strict=True):
        ax = sample.plot(
            method="hist",
            ax=ax,
            clear=ax is None,
            bins=bins,
            density=True,
            histtype="step",
            label=label,
            show=False,
        )
    _ = ax.legend()
scp.show()

# %%
# Average the spectra over the distribution
# -----------------------------------------
# This small node count keeps the example quick. Increase both ``samples``
# and powder ``accuracy`` when checking convergence for an actual fit.
spectra = []
labels = []
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
    spectrum.x.ito("kHz")
    spectra.append(spectrum.real)
    labels.append(label)

_ = scp.plot_multiple(
    spectra,
    method="pen",
    labels=labels,
    legend="best",
    figsize=(8, 3.4),
    xlim=(1, 7),
    ylabel="Bin mass",
)
