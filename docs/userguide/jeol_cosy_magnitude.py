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
# JEOL COSY in magnitude mode
# =============================
#
# Reconstruct a complete 2D proton COSY spectrum from a uniform raw JEOL
# JDF. The recipe applies a squared sine window to each time dimension,
# performs both Fourier transforms, takes the magnitude and symmetrizes
# the spectrum. The result can be plotted directly with frequency axes
# in both dimensions.
#
# This example uses the epicatechin COSY from
# `CENAPTNMR, nmrXiv P33 <https://doi.org/10.57992/nmrxiv.p33>`_, released
# under CC0 1.0. The documentation fixture manifest pins the file's source
# revision, size and SHA-256. Run ``python docs/fetch_gallery_data.py``
# from the plugin checkout to fetch the example data. The example uses
# the development plugin after 0.4.0 with SpectroChemPy 1.3.1 and
# nmrglue-ng 1.3.0 or later.
#
# From raw JDF to a 2D spectrum
# --------------------------------
#
# Use ``process_jeol_cosy_magnitude()`` for supported uniform proton COSY
# acquisitions recorded with ``cosy.jxp`` or ``cosy_pfg.jxp``. Supply the
# raw JDF directly. The function handles reading and both transforms.

# %%
import spectrochempy as scp
from spectrochempy_nmr import process_jeol_cosy_magnitude

path = scp.preferences.datadir / "nmrdata/jeol/epicatechin_cosy.jdf"
cosy = process_jeol_cosy_magnitude(path, f1_size=2048)
print("Acquired shape:", cosy.meta.raw_shape)
print("Processed shape:", cosy.shape)
print("Frequency axes F1, F2:", cosy.y.units, cosy.x.units)
_ = cosy.plot(
    method="contour",
    start=0.02,
    nlevels=10,
    negative=False,
    cmap="viridis",
    figsize=(7, 6),
    title="Epicatechin: complete 2D COSY magnitude spectrum",
)

# %% [raw] raw_mimetype="text/restructuredtext"
# The input contains 256 F1 increments with 1280 direct samples each.
# Both output axes are in Hz. Here ``f1_size=2048`` zero-fills F1 to
# 2048 points; F2 is zero-filled to the next power of two, also 2048.
# Zero filling makes a denser display grid without adding acquired
# resolution. Omitting ``f1_size`` retains the acquired F1 size.
#
# Read the cross-peaks
# -------------------------
#
# Diagonal peaks have the same proton frequency in F1 and F2. The pairs
# away from the diagonal show proton correlations. Inspecting a smaller
# region helps separate these cross-peaks from the strongest diagonal
# signals. The plot below uses the same processed spectrum, without
# applying an additional transform or normalization.

# %%
_ = cosy.plot(
    method="contour",
    start=0.005,
    nlevels=10,
    negative=False,
    cmap="viridis",
    xlim=(2100, 2500),
    ylim=(2100, 2500),
    figsize=(6, 6),
    title="Epicatechin: COSY cross-peaks",
)

# %% [raw] raw_mimetype="text/restructuredtext"
# The correlation near F1 = 2270 Hz and F2 = 2338 Hz is one of the
# positions checked against the processed epicatechin export in the
# qualification tests. The axes use the frequency calibration stored
# in the JDF. They are displayed in Hz; this workflow does not require
# a manual ppm reference.
#
# What the processing applies
# --------------------------------
#
# The squared sine windows taper each acquired FID to zero at both ends.
# The two-dimensional FFT is followed by magnitude, producing a real,
# nonnegative spectrum. Magnitude removes phase and sign information.
#
# Minimum symmetrization evaluates the spectrum at exchanged physical
# frequencies and retains the smaller amplitude in each pair. This
# suppresses asymmetric features and can change relative peak heights.
# Points outside the overlap of the two frequency ranges become zero.
# Use this spectrum to inspect correlations; peak magnitudes should not
# be treated as quantitative integrals without separate validation.
#
# In the qualification suite, processed COSY exports for epicatechin,
# luteolin and rutin serve as comparisons for positions and relative
# intensities. For a direct comparison with an export on another grid,
# pass its F1 and F2 coordinate arrays as
# ``frequency_grid=(reference.y.data, reference.x.data)``. The function
# interpolates the magnitude before applying the symmetric minimum.
#
# For a different raw file, confirm that it is a supported uniform
# proton COSY acquisition before using this recipe. See
# :doc:`../reference/core_operations` for the function parameters and
# :doc:`reading_data` for the catalogue of supported input types.
