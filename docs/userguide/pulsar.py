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
# # Simulate and fit powder spectra with PULSAR
#
# PULSAR simulates one-dimensional solid-state NMR powder spectra for
# quadrupolar transitions and chemical shift anisotropy (CSA). It supports static
# powder, magic-angle spinning and variable-angle spinning. Finite RF
# sequences can include secular isotropic J coupling.
#
# ## Ideal-pulse spectra
#
# For a spin-5/2 central transition with two sidebands on each side,
# use `simulate_central_transition()`.

# %%
import spectrochempy as scp
from spectrochempy.nmr import pulsar

from spectrochempy_nmr.pulsar import Delay, IdealPulse

spectrum = pulsar.simulate_central_transition(
    spin=2.5,
    cq_hz=4e6,
    eta=0.6,
    larmor_hz=100e6,
    spectral_width_hz=100e3,
    shift_hz=4500,
    spinning_hz=10e3,
    sidebands=2,
)

spectrum.x.ito("kHz")
_ = spectrum.plot(method="pen", figsize=(8, 3), ylabel="Intensity per bin")

# %% [markdown]
# The returned dataset has an increasing frequency offset axis in Hz.
# Each intensity is the mass in a bin whose left edge is the coordinate.
# The selected bands have unit total area. Choose a simulation window
# wide enough to contain all selected bands and transitions.
#
# Use `simulate_quadrupolar()` for satellites
# or all transitions. This includes first- and second-order quadrupolar
# effects and supports integer and half-integer spins. For CSA alone,
# use `simulate_csa()`. Its signed anisotropy
# is the principal value along z minus the isotropic shift, in Hz.
# To combine CSA and quadrupolar effects, supply the CSA anisotropy,
# asymmetry and optional relative rotation to either quadrupolar
# function. The columns of the rotation matrix give the CSA principal
# axes in the electric-field-gradient frame.
#
# ## Preparation and detection
#
# `simulate_pulse_sequence()` accepts an
# ordered list of finite or ideal RF pulses, delays and coherence
# selections. Use `simulate_coupled_sequence()`
# for two RF-active spins coupled by isotropic J. Dipolar coupling is
# outside the model. For example, this static spin-1/2 echo uses the
# receiver phase to place the refocused signal in positive absorption.

# %%
echo = pulsar.simulate_pulse_sequence(
    [
        IdealPulse(90),
        Delay(200e-6),
        IdealPulse(180),
        Delay(200e-6),
    ],
    spin=0.5,
    larmor_hz=100e6,
    spectral_width_hz=20e3,
    csa_anisotropy_hz=3000,
    csa_eta=0.5,
    receiver_phase_deg=180,
    t2_s=1e-3,
)

echo.x.ito("kHz")
_ = scp.plot_multiple(
    [echo.real, echo.imag],
    method="pen",
    labels=["Absorption", "Dispersion"],
    legend="best",
    figsize=(8, 3),
    ylabel="Intensity per bin",
)

# %% [markdown]
# Sequence output is complex and retains excitation losses. Its amplitude
# scale comes from an ideal on-carrier 90° x pulse. Pass
# `output="density"` to obtain the powder-averaged deviation density
# matrix at the end of preparation.
#
# ## Fit a central transition
#
# Pass a PULSAR line shape to `scp.Optimize` as a user model. The model
# returns intensities on the observed frequency grid. In this example,
# the frequency coordinate contains offsets in Hz.
#
# The input is a simulated central transition with Gaussian broadening
# of 115 Hz FWHM applied by `pulsar.broaden()`.

# %%
simulation = {
    "spin": 2.5,
    "larmor_hz": 100e6,
    "spectral_width_hz": 20e3,
    "spinning_hz": 8e3,
    "points": 4096,
    "accuracy": 8,
}
simulated = pulsar.simulate_central_transition(
    cq_hz=2.6e6, eta=0.55, shift_hz=725, **simulation
)
half_bin = (simulated.x[1] - simulated.x[0]) / 2
observed = pulsar.broaden(simulated, gaussian_fwhm_hz=115)
observed.set_coordset(x=simulated.x + half_bin)
observed = observed[-4000.0:3500.0]


def central_transition(x, ampl, cq, eta, shift, fwhm):
    spectrum = pulsar.simulate_central_transition(
        cq_hz=cq, eta=eta, shift_hz=shift, **simulation
    )
    spectrum = pulsar.broaden(spectrum, gaussian_fwhm_hz=fwhm)
    spectrum.x += half_bin
    return ampl * spectrum.interpolate(dim="x", coord=x, fill_value=0).data


fit = scp.Optimize(
    usermodels={"pulsar": central_transition},
    amplitude_mode="area",
)
fit.script = """
MODEL: central
shape: pulsar
$ ampl: 1, 0, 3
$ cq: 2500000, 2000000, 3000000
$ eta: 0.5, 0.05, 0.95
$ shift: 680, 0, 1500
$ fwhm: 100, 30, 220
"""
result = fit.fit(observed).result

for dataset in (observed, result.fitted, result.residuals):
    dataset.x.ito("kHz")

_ = scp.plot_multiple(
    [observed, result.fitted],
    method="pen",
    labels=["Simulated input", "Fit"],
    legend="best",
    ls=["-", "--"],
    figsize=(8, 3),
    xlim=(-4, 3.5),
    ylabel="Intensity per bin",
)
_ = result.residuals.plot(
    method="pen", figsize=(8, 2), xlim=(-4, 3.5), ylabel="Residual"
)

# %% [markdown]
# The fit varies the quadrupolar coupling, asymmetry, isotropic shift,
# Gaussian width and a nonnegative amplitude. The script expresses
# `cq`, `shift` and `fwhm` in Hz. Read the optimized values from
# `fit.script` and the spectra and fit diagnostics from `fit.result`.
# Choose bounds that keep the selected transitions inside the simulation
# window. Repeat the fit with a finer powder mesh to check stability.
#
# ## Scope of the checks
#
# Spin-Hamiltonian and time-domain calculations check selected
# static and spinning cases. The experimental aluminum acetylacetonate
# check covers one central band. In the experimental YAG comparison,
# predicted sideband peaks agree within one measured bin. The areas
# still show larger differences, so the YAG check does not establish
# quantitative experimental intensity accuracy.

# %% [raw] raw_mimetype="text/restructuredtext"
# Historical notes
# ----------------
#
# * The original PULSAR paper is :cite:p:`amoureux1995pulsar`.
# * See :cite:t:`masai` for the archived software.
