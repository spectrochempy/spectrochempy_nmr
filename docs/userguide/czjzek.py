# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Czjzek distributions of quadrupolar tensors
#
# The NMR module can average powder spectra over an electric-field-gradient
# (EFG) distribution. The standard Czjzek model describes isotropic Gaussian
# disorder in the five independent coordinates of a symmetric traceless
# tensor. The extended model adds that disorder to a persistent local tensor.
#
# ## Construct a model and simulate its powder spectrum
#
# Compare the extended distribution with its local tensor at the same
# isotropic shift. Both spectra have unit total area.

# %%
import spectrochempy as scp

distribution = scp.nmr.czjzek.CzjzekDistribution(
    sigma_hz=0.3e6,
    cq_hz=3e6,
    eta=0.6,
    shift_hz=5000,
    shift_sigma_hz=200,
)
settings = dict(
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

single_tensor = scp.nmr.czjzek.CzjzekDistribution(
    sigma_hz=0,
    cq_hz=distribution.cq_hz,
    eta=distribution.eta,
    shift_hz=distribution.shift_hz,
)

spectra = [
    scp.nmr.czjzek.simulate_czjzek(model, **settings)
    for model in (single_tensor, distribution)
]
for spectrum in spectra:
    spectrum.x.ito("kHz")

_ = scp.plot_multiple(
    spectra,
    method="pen",
    labels=["Local tensor", "Extended Czjzek"],
    legend="best",
    figsize=(8, 3),
    xlim=(-3, 8),
    ylabel="Intensity per bin",
)

# %% [markdown]
# The distribution broadens the central line around the same isotropic shift.
# `cq_hz` and `eta` define the local tensor. `sigma_hz` is the standard
# deviation of each orthonormal EFG coordinate in frequency units. Set
# `cq_hz=0` for the standard Czjzek model. `shift_sigma_hz` adds an
# independent Gaussian distribution of isotropic shifts. Frequency
# parameters are in Hz.
#
# A fixed `seed` makes the simulation reproducible. Increase `samples`
# and powder `accuracy` separately to check convergence.
#
# ## Experimental limits
#
# The public PBG1 benchmark leaves an error on satellite-band areas after
# constraining the distribution with 3QMAS. The inferred disorder and
# predicted intensities are sensitive to sampling and assumptions about RF
# transfer. See the PBG1 benchmark in
# `src/spectrochempy_nmr/README.pulsar.md` for the protocol and comparisons.

# %% [raw] raw_mimetype="text/restructuredtext"
# References
# ----------
#
# The standard and extended models are described in
# :cite:p:`despinose2008czjzek,lecaer2010czjzek`.
