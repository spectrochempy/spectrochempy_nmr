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
# Build a DOSY map
# ================
#
# Display where signals appear in the spectrum and how quickly they diffuse.
# The horizontal axis is chemical shift; the vertical axis is diffusion D.
# Smaller D means slower motion.
#
# Open and process
# ----------------
#
# Run this block with an internet connection for the first download.
# It opens the PEG 600 example, reduces noise with a light 1 Hz
# exponential apodization in F2, and applies one phase to the whole series.

# %%
import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/relax/102/ser"
diffusion = scp.nmr.Diffusion(path)
spectra = diffusion.process(lb=1.0)
_ = spectra[:, 74.0:69.0].plot(method="lines", colorbar=True)

# %% [raw] raw_mimetype="text/restructuredtext"
# Display the DOSY map
# --------------------
#
# Select the spectral region in ppm and the range of D in m²/s.

# %%
result = diffusion.dosy(region=(74.0, 69.0), D=(1e-12, 5e-11))
_ = result.plot()

# %% [raw] raw_mimetype="text/restructuredtext"
# The PEG signals should appear at a similar height, around
# **8 × 10⁻¹² m²/s**. This sample illustrates a map; for signals from
# different species, different heights can reveal different motion.
# Map widths and extra spots depend on noise and processing, so they
# should not be read directly as a count of molecules.
# The display hides intensities below 5% of the strongest map signal;
# use ``result.plot(start=0.01)`` to show weaker features.
#
# Check the result
# ----------------
#
# Display the difference between the measured and reconstructed spectra.
# Peaks that remain in this difference need attention before interpreting
# the map. The reconstructed series is also available as ``result.fitted``.

# %%
_ = result.residual.plot(method="lines", colorbar=True)

# %% [raw] raw_mimetype="text/restructuredtext"
# Adapt and save
# --------------
#
# * Change ``region`` to show your peaks, in ppm.
# * Adjust ``lb`` in ``process(lb=1.0)`` to change the smoothing, in Hz.
#   A larger value reduces noise but broadens peaks; use ``lb=0`` to remove it.
# * Change ``D`` to set the vertical range, in m²/s. If signal reaches the
#   top or bottom edge, inspect the data and try a wider range.
# * Replace ``path`` with your supported Bruker ``diffSte`` ``ser`` file,
#   keeping ``diff.xml`` and the parameter files beside it.
#
# To save the map, run ``result.map.write("dosy.scp")`` in your notebook.
# For the inversion settings and checks, see the
# :doc:`diffusion appendix <diffusion_details>`.
#
# File: ``nmrdata/bruker/tests/nmr/relax/102/ser``
#
# Author: Cassandre Kouvatas · Licence: CC BY 4.0.
