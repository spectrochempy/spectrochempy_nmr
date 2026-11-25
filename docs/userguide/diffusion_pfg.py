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

# %% [raw] raw_mimetype="text/restructuredtext"
# Measure a diffusion coefficient
# ===============================
#
# Measure how quickly molecules move in solution. In this PFG experiment,
# the signal becomes weaker as the diffusion weighting increases. Fit that
# decrease to estimate a diffusion coefficient, called D.
#
# Start with the public carbon-13 PEG 600 example below. You do not need to
# type gradients, calculate diffusion weights or write a fitting function.
#
# Open and display the series
# ---------------------------
#
# Keep an internet connection for the first read, which may download the
# example. Run this block, then inspect the spectra around 73 ppm.

# %%
import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/relax/102/ser"
diffusion = scp.nmr.Diffusion(path)
spectra = diffusion.process()
_ = spectra[:, 74.0:71.0].plot(method="lines", colorbar=True)

# %% [raw] raw_mimetype="text/restructuredtext"
# The peaks should become smaller across the series. The same phase is
# applied to every spectrum. Check the line shape and baseline before
# measuring an area; keep the relative signal strengths between spectra.
#
# Measure one peak
# ----------------
#
# Select the peak between 72.649 and 72.916 ppm. The fit compares its area
# across the series and displays the measured curve, the fitted curve and
# their difference below it.

# %%
result = diffusion.fit(region=(72.649, 72.916))
_ = result.plot()
result.D

# %% [raw] raw_mimetype="text/restructuredtext"
# You should obtain D close to **7.6 × 10⁻¹² m²/s**. A smaller D describes
# slower motion. The residual shows what the fitted curve does not explain.
# A structured residual, overlapping peaks or a changing baseline can make a
# single fitted coefficient misleading.
#
# Adapt and save
# ---------------
#
# * Change ``region`` to measure another isolated peak, with limits in ppm.
# * For your own supported Bruker ``diffSte`` acquisition, replace ``path``
#   with its ``ser`` file and keep ``diff.xml`` and the parameter files beside it.
# * To adjust the common phase, call ``diffusion.process(phc0=..., phc1=...)``
#   with phase values in degrees, then repeat the measurement block.
#
# To save the measured curve, run
# ``result.areas.write("diffusion-areas.scp")`` in your notebook. Use
# ``result.fitted`` and ``result.residual`` to inspect or save the comparison.
#
# This recipe estimates one apparent D for the selected signal. It does not
# produce a DOSY map or separate several diffusion components. For supported
# inputs and other prepared series, see :doc:`/reference/diffusion`.
# Continue with :doc:`diffusion_dosy` to display a diffusion map.
# The :doc:`diffusion appendix <diffusion_details>` contains the reference
# comparison and measurement conventions.
#
# File: ``nmrdata/bruker/tests/nmr/relax/102/ser``
#
# Author: Cassandre Kouvatas · Licence: CC BY 4.0.
