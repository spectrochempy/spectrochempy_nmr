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
# sequences can include secular isotropic J coupling and, for
# heteronuclear spin-½ pairs, static or MAS dipolar coupling and cross polarization.
#
# %% [raw] raw_mimetype="text/restructuredtext"
# See :doc:`pulsar_scope` for the history, model limits, and comparisons
# with other simulators.

# %% [markdown]
# ## ²⁷Al in Al(acac)₃ at two MAS rates
#
# For the spin-5/2 central transition, the first-order quadrupolar shift
# vanishes. The second-order interaction still broadens and shifts the
# line under MAS. At fixed spin and asymmetry its frequency scale is
# proportional to Cq²/ν₀, where ν₀ is the Larmor frequency.
#
# Simulate one site with the same tensor and field at 0.5 and 1 kHz.
# The model uses ideal excitation at the magic angle. Both profiles
# receive the same 65 Hz Gaussian broadening for display.

# %%
import spectrochempy as scp
from spectrochempy.nmr import pulsar

from spectrochempy_nmr.pulsar import Delay, IdealPulse

settings = dict(
    spin=2.5,
    cq_hz=2.923e6,
    eta=0.136,
    larmor_hz=104.46851646e6,
    spectral_width_hz=48e3,
    points=8192,
    rotor_angle_deg=54.735610317,
    accuracy=24,
    sidebands=20,
)

spectra = [
    pulsar.broaden(
        pulsar.simulate_central_transition(spinning_hz=rate, **settings),
        gaussian_fwhm_hz=65,
    )
    for rate in (500, 1000)
]
for spectrum in spectra:
    spectrum.x.ito("kHz")

_ = scp.multiplot(
    spectra,
    labels=["Simulated MAS at 0.5 kHz", "Simulated MAS at 1 kHz"],
    ncol=2,
    sharex=True,
    sharey=True,
    method="pen",
    figsize=(9, 3),
    xlim=(-8, 8),
    xlabel="Frequency offset (kHz)",
    ylabel="Intensity per bin",
)

# %% [markdown]
# Sideband orders are separated by the rotor frequency. At 0.5 kHz
# they lie closer together and carry more of the total intensity.
# The second-order shift moves the central band even though the
# isotropic chemical-shift offset is set to zero.
#
# PULSAR returns frequency offsets in Hz at the left edges of the bins.
# Each intensity is a bin mass. The masses sum to one across the selected
# orders before broadening. Check the window, number of orders and powder
# mesh before comparing relative intensities. Use the physical Larmor
# frequency for the quadrupolar calculation and apply the chemical-shift
# reference separately when aligning to a measured spectrum.
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
# for two RF-active spins coupled by isotropic J, optionally with a
# heteronuclear spin-½ dipolar interaction. This spin-1/2 echo uses the
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
# ## Dipolar coupling and cross polarization
#
# Add `dipolar_hz=5000` for a heteronuclear spin-½ pair. The
# interaction in Hz is `D * (3*cos(theta)**2 - 1) * Iz*Sz`, using the
# historical PyPulsar convention. `dipolar_axis` gives a unit bond vector
# in the common sample frame, defaulting to `(0, 0, 1)`. Both pulse
# propagation and acquisition include the orientation-dependent coupling.
# In static samples, simultaneous RF contacts transfer magnetization near
# the Hartmann–Hahn condition, where the two RF amplitudes match.
#
# Set `spinning_hz` for MAS. The rotor modulation continues during RF,
# delays and acquisition, producing dipolar sidebands. Under MAS the
# first-order RF difference or sum matches one or two rotor frequencies.
# The rank-two mean vanishes at the exact magic angle, but the modulation
# still affects RF transfer and acquisition. Select enough `sidebands`
# for spectral output, and converge `accuracy` and `max_rf_step_s`.
#
# The model describes an isolated pair, with ¹H–¹³C as one example rather
# than an isotope restriction. It does not include homonuclear flip-flop terms,
# a proton bath, spin diffusion or rotating-frame relaxation.
# `Delay(decouple_j=True)`
# removes J only; it leaves the dipolar interaction active.
#
# %% [raw] raw_mimetype="text/restructuredtext"
# The :doc:`static CP example
# <../examples/gallery/simulation/plot_pypulsar_cp_static>` shows contact-time
# and Hartmann–Hahn curves with a two-scan cycle that isolates transferred
# proton magnetization from directly excited carbon signal.
# The :doc:`MAS CP example
# <../examples/gallery/simulation/plot_pypulsar_cp_mas>` compares the first
# rotor-sideband match with equal RF fields and sweeps the RF matching profile.

# %% [markdown]
# ## Fit a central transition
#
# Pass a PULSAR line shape to `scp.Optimize` as a user model. The model
# returns intensities on the observed frequency grid. In this example,
# the frequency coordinate contains offsets in Hz.
#
# The input is a simulated central transition with Gaussian broadening
# of 115 Hz FWHM applied by `pulsar.broaden()`. The half-bin offset moves
# the model coordinates from the left edges to the bin centers before
# interpolation onto the observation grid.

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
# window. This synthetic fit checks parameter recovery with the same
# model used to generate the input. For measured data, inspect the
# residuals and repeat the fit with different starting values and a
# finer powder mesh before interpreting the parameters.
#
# ## Scope of the checks
#
# Spin-Hamiltonian and time-domain calculations check selected
# static and spinning cases. The experimental aluminum acetylacetonate
# check covers one central band. In the experimental YAG comparison,
# predicted sideband peaks agree within one measured bin. The areas
# still show larger differences, so the YAG check does not establish
# quantitative experimental intensity accuracy.
