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
# # Compare a measured alumina spectrum with a disorder model
#
# The archived experiment is titled "Alumine 800 C" and observes 27Al at
# 130.306 MHz. It is distributed in `spectrochempy_data` under CC BY 4.0,
# supplied with permission by Christian Fernandez. This guide reads the
# existing processed spectrum and overlays an illustrative two-component
# extended Czjzek model. It does not adjust that model to the observation.
#
# ## Read and select the central region
#
# Use the usual SpectroChemPy `datadir`. The documentation fixture fetch
# includes the processed real spectrum and its acquisition/processing
# parameters. `proc_axis=True` retains the archived processing reference.
# The explicit selection below is in ppm; normalize only this displayed
# region so that a distant sideband does not set the plot scale.

# %%
import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/alumina_800C/12/pdata/1/1r"
observed = scp.nmr.read(path, proc_axis=True).squeeze().real
observed = observed[100.0:-35.0]
observed = observed / observed.max()

_ = observed.plot(
    method="pen",
    figsize=(8, 3),
    xlim=(-35, 100),
    reverse=True,
    xlabel="Chemical shift / ppm",
    ylabel="Peak-normalized intensity",
)

# %% [markdown]
# ## Construct a two-environment illustration
#
# Two local tensors at 13 and 75 ppm illustrate octahedral-like and
# tetrahedral-like regions. These labels and parameters are illustrative;
# this plot alone does not establish site populations or assignments.
# Each ideal profile has unit area before mixing. The coefficients 0.65
# and 0.35 are assumed model weights, not measured concentrations.
#
# The disorder widths are 0.9 and 1.2 MHz in the EFG-coordinate convention
# of `sigma_hz`, with local couplings of 4.5 and 5.5 MHz. We additionally
# apply a Gaussian broadening of 800 Hz FWHM (about 6.1 ppm at this field)
# to each simulated spectrum before converting its axis to ppm. This is a
# phenomenological spectral width, distinct from the Czjzek tensor disorder.
# It rounds the sharp powder-profile edges; it is not an independently
# determined lifetime or chemical-shift distribution. These manually chosen
# parameters illustrate bands on the scale of the observed widths. They
# have not been optimized against the measurement.
#
# The simulation uses 12 kHz MAS, close to the archived sideband spacing.
# Archived `MASR=5 kHz` and `CNST31=12.5 kHz` disagree. Resolve the actual
# spinning rate before quantitative interpretation of sideband intensities.
# This illustration only shows the central transition.

# %%
larmor_hz = 130.30630648e6
models = (
    (
        "Octahedral model",
        scp.nmr.czjzek.CzjzekDistribution(
            0.9e6, cq_hz=4.5e6, eta=0.5, shift_hz=13e-6 * larmor_hz
        ),
    ),
    (
        "Tetrahedral model",
        scp.nmr.czjzek.CzjzekDistribution(
            1.2e6, cq_hz=5.5e6, eta=0.5, shift_hz=75e-6 * larmor_hz
        ),
    ),
)

profiles = []
labels = []
for label, distribution in models:
    spectrum = scp.nmr.czjzek.simulate_czjzek(
        distribution,
        samples=128,
        seed=0,
        spin=2.5,
        larmor_hz=larmor_hz,
        spinning_hz=12e3,
        spectral_width_hz=200e3,
        points=4097,
        accuracy=4,
        transitions="central",
    )
    spectrum = scp.nmr.pulsar.broaden(spectrum, gaussian_fwhm_hz=800)
    spectrum.x.meta["acquisition_frequency"] = larmor_hz * scp.ur.Hz
    spectrum.x.ito("ppm")
    profiles.append(spectrum.real)
    labels.append(label)

mixture = 0.65 * profiles[0] + 0.35 * profiles[1]

_ = scp.plot_multiple(
    [*profiles, mixture],
    labels=[*labels, "Assumed mixture"],
    method="pen",
    legend="best",
    figsize=(8, 3),
    xlim=(-35, 100),
    reverse=True,
    xlabel="Chemical shift / ppm",
    ylabel="Intensity per bin",
)

# %% [markdown]
# Convert the simulated frequency axes using the observation frequency
# before plotting in ppm. Only the final overlay is peak normalized;
# this deliberately removes the absolute excitation and concentration scale.

# %%
_ = scp.plot_multiple(
    [observed, mixture.normalize()],
    labels=["Alumine 800 C, measured", "Illustrative mixture"],
    method="pen",
    legend="best",
    figsize=(8, 3.8),
    xlim=(-35, 100),
    reverse=True,
    xlabel="Chemical shift / ppm",
    ylabel="Peak-normalized intensity",
)

# %% [markdown]
# ## What to check before fitting measured data
#
# The overlay is a visual comparison. Differences can reflect assumed
# disorder, isotropic shift distribution, RF excitation, phase, baseline,
# referencing or receiver response. A visually plausible mixture does not
# validate the assumed tensor parameters.
#
# For an actual adjustment, define the fitted spectral window and residual,
# use consistent normalization and response settings, check ensemble and
# powder convergence, and inspect parameter correlations. Start with the
# synthetic fitting workflow, where the injected width
# and all other parameters are known.
#
# %% [raw] raw_mimetype="text/restructuredtext"
# See also
# --------
#
# * :doc:`czjzek_fit`: synthetic fitting with known parameters.
# * :doc:`Short alumina recipe </examples/gallery/simulation/plot_alumina_czjzek>`.
