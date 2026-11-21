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
# # Build and fit a finite-RF Czjzek library
#
# A finite RF pulse weights the complex response of each tensor in a Czjzek
# ensemble. Repeating that propagation at every optimizer evaluation can be
# expensive. `CzjzekRFLibrary` stores responses on a signed coupling and
# asymmetry grid for one fixed pulse sequence, spectral grid and receiver.
# It interpolates covered tensors and calculates uncovered members directly.
# `fit_czjzek` can use this approximation for initialization, then always
# refines the parameters with direct simulation.
#
# This worked example estimates one disorder width from a synthetic spectrum.
# The local coupling, asymmetry, shift, pulse and receiver are known and fixed.
# It demonstrates the workflow, not identification of all those parameters
# from a single measured spectrum.
#
# ## Define the finite-pulse model and library grid
#
# The nucleus has spin 5/2, at 130 MHz, with a 0.5 microsecond pulse at
# 50 kHz RF amplitude. MAS defaults to zero here: this small static example
# keeps execution short. A MAS library additionally fixes `spinning_hz`,
# `rotor_angle_deg` and `sidebands`. All frequencies below are in Hz.

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

# %% [markdown]
# The signed `cq_grid_hz` covers both signs that can occur in sampled EFG
# tensors. `eta_grid` covers asymmetries from zero to one. These are tensor
# grids, not grids of the distribution parameter `sigma_hz`.
#
# With 513 points over 20 kHz, the bin spacing is 20000/512 Hz.
# `shift_bins=8` covers shifts within 312.5 Hz of `shift_hz=0`.
# Building the library simulates two displacement planes for every coupling
# and asymmetry pair. The spectral window must contain the transitions,
# including room for that alignment. A wider shift range is not a replacement
# for a sufficiently wide spectral window.
#
# ## Save once and validate the model when reloading
#
# A temporary directory keeps this executable example self-contained. For
# reuse between sessions, choose a persistent output directory instead.
# Loading requires the same sequence, grids and simulation settings; a
# mismatch raises `ValueError` rather than silently reusing another model.

# %%
with TemporaryDirectory() as directory:
    path = Path(directory) / "czjzek-rf.npz"
    library.save(path)
    library = scp.nmr.CzjzekRFLibrary.load(path, events, **grid, **settings)

# %% [markdown]
# The archive contains complex responses and model metadata. Changing RF
# amplitude, duration, phase cycle, spectral grid, MAS settings or
# `probe_bandwidth_hz` requires a matching library. The receiver bandwidth
# must also be passed unchanged to the direct refinement; see the
# receiver discussion in the distribution guide.
#
# ## Generate a synthetic observation
#
# The injected disorder width is 150 kHz. `sigma_hz` is the standard deviation
# of each orthonormal EFG coordinate; it is not a line-width FWHM or a Cq
# standard deviation. A 400 Hz Gaussian FWHM is applied identically to every
# prediction and the observation. This fixed spectral broadening is distinct
# from tensor disorder. We fit the real component with the simulated phase
# retained, without peak normalization at each trial.


# %%
def model(sigma_khz):
    return scp.nmr.czjzek.CzjzekDistribution(sigma_khz * 1e3, cq_hz=2e6, eta=0.4)


def processed(spectrum):
    return scp.nmr.pulsar.broaden(spectrum, gaussian_fwhm_hz=400).real


observed = processed(
    scp.nmr.czjzek.simulate_czjzek(
        model(150), events=events, samples=16, seed=7, **settings
    )
)

interpolated = processed(library.simulate(model(150), samples=16, seed=7))
_ = scp.plot_multiple(
    [observed, interpolated, observed - interpolated],
    labels=["Direct observation", "Interpolated", "Interpolation residual"],
    method="pen",
    legend="best",
    figsize=(8, 3),
    xlim=(-3000, 1500),
)

# %% [markdown]
# This comparison isolates interpolation error at the known parameters.
# Increase the coupling and asymmetry grid densities separately to check it.
# The small example uses 16 Sobol ensemble members and powder accuracy 2;
# increase those independently for quantitative work. Keep the same `seed`
# and `samples` for the observation and both fitting stages so that changing
# quadrature samples does not add a moving numerical target to the objective.
#
# ## Define one objective for both stages
#
# The callback receives the parameter vector and a simulation callable.
# `fit_czjzek` supplies either the library evaluator or the direct evaluator.
# The objective uses the squared residual relative to the observation energy.
# The parameter is expressed in kHz to keep the numerical scale convenient,
# with bounds 80 to 220 kHz and an initial estimate of 110 kHz.


# %%
def objective(parameters, simulate):
    residual = processed(simulate(model(parameters[0]))) - observed
    return float((residual**2).sum()) / float((observed**2).sum())


result = scp.nmr.fit_czjzek(
    objective,
    [110],
    [(80, 220)],
    events=events,
    library=library,
    samples=16,
    seed=7,
    method="Powell",
    options={"xtol": 0.01, "ftol": 1e-6, "maxiter": 12},
    **settings,
)
sigma = result.direct.optimization.x[0]
fitted = processed(
    scp.nmr.czjzek.simulate_czjzek(
        model(sigma), events=events, samples=16, seed=7, **settings
    )
)
print(
    f"Direct fit: sigma = {sigma:.2f} kHz; success = {result.direct.optimization.success}"
)

# %% [markdown]
# Always read final parameters and optimizer status from `result.direct`.
# `result.interpolated` records the initialization stage, when a library was
# supplied. Both stages expose `optimization`, elapsed `seconds` and exact
# `propagations` (including direct fallbacks during interpolation).
# Remove `library=library` to run a direct-only fit with the same objective.

# %%
print(f"Interpolated initialization: {result.interpolated.optimization.x[0]:.2f} kHz")
print(f"Relative squared residual: {result.direct.optimization.fun:.3g}")
_ = scp.plot_multiple(
    [observed, fitted, observed - fitted],
    labels=["Synthetic observation", "Direct fit", "Residual"],
    method="pen",
    legend="best",
    figsize=(8, 3),
    xlim=(-3000, 1500),
)

# %% [markdown]
# The final direct result recovers the injected 150 kHz width in this
# noise-free, matched-model example. This does not establish uncertainty or
# uniqueness for measured data. With real spectra, check phase, baseline,
# frequency referencing, receiver response, RF transfer and parameter
# correlations before interpreting an adjusted distribution.
#
# ## Accuracy, coverage and cost
#
# - Tensors outside the coupling/asymmetry grids, or outside the displacement
#   range, are propagated directly. These fallbacks preserve the requested
#   model but can remove the anticipated acceleration.
# - A covered tensor can still have transitions outside the spectral window;
#   enlarge the window instead of accepting a clipped prediction.
# - A denser interpolation grid does not improve ensemble or powder sampling.
#   Compare direct and interpolated spectra after the same processing.
# - Library construction, storage and loading have a cost. Repeated fits may
#   amortize construction; this example makes no general speedup claim.
# - A successful optimizer is not a convergence study. Check sampling, grid
#   density, starting values, bounds and residual structure separately.
#
# %% [raw] raw_mimetype="text/restructuredtext"
# See also
# --------
#
# * :doc:`czjzek`: distribution parameters and receiver response.
# * :doc:`Save/reload a library </examples/gallery/simulation/plot_czjzek_library>`.
# * :doc:`Fit a disorder width </examples/gallery/simulation/plot_fit_czjzek>`.
