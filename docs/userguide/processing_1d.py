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
# Turn a FID into a 1D spectrum
# =============================
#
# A FID is the signal recorded after excitation. This recipe turns a public
# Bruker proton FID into a spectrum with chemical shift in ppm. Run the
# cells in order in the notebook opened during
# :doc:`installation </getting_started/installation>`.
#
# Read, process and display
# -------------------------
#
# The settings below belong to this example. The phase angles make its
# resonances point upwards; use angles chosen for your own data when you
# replace the file. ``remove_digital_filter=True`` corrects the Bruker
# receiver delay while reading, so it is not corrected again below.

# %%
import spectrochempy as scp

fid = scp.nmr.read(
    "nmrdata/bruker/tests/nmr/topspin_1d/1/fid",
    remove_digital_filter=True,
    remove_dc_offset=False,
)
spectrum = scp.nmr.Experiment(fid).process(
    apodization="em",
    lb=2.0,
    size=32768,
    phase="manual",
    phc0=52.43836,
    phc1=-16.8366,
)
_ = spectrum.real.plot(xlim=(-2.0, 12.0))

# %% [raw] raw_mimetype="text/restructuredtext"
# You should see narrow resonances on a ppm axis. Before measuring areas,
# check that the peaks have the expected sign and shape and that the line
# between them is flat. The original ``fid`` is still available.
#
# Settings you can change
# -----------------------
#
# .. list-table::
#    :header-rows: 1
#
#    * - Setting
#      - Meaning
#    * - The quoted file path
#      - Replace it with your own Bruker FID path; see :doc:`reading_data`.
#    * - ``lb=2.0``
#      - Line broadening in Hz. More smoothing also makes peaks wider.
#    * - ``size=32768``
#      - Number of points in the displayed spectrum. More points do not
#        recover experimental resolution that was not acquired.
#    * - ``phc0``, ``phc1``
#      - Phase angles in degrees. These two values apply to this example only.
#    * - ``xlim=(-2.0, 12.0)``
#      - The displayed ppm range. Changing it does not remove data.
#
# Remove a baseline and measure a peak
# ------------------------------------
#
# For this example's positive peaks, estimate the baseline, select the main
# resonance between -0.1 and 1.5 ppm, and integrate it. Change these limits
# for another resonance. The area is in intensity units times ppm; it is
# not automatically a concentration.

# %%
corrected = spectrum.real.snip()
_ = corrected.plot(xlim=(-0.5, 2.0))
area = corrected[-0.1:1.5].trapezoid()
area

# %% [raw] raw_mimetype="text/restructuredtext"
# Check the corrected line between peaks and whether the integration range
# includes the whole peak without its neighbors. See :doc:`baseline` for
# other baseline methods, especially when peaks can be negative.
#
# Save and continue
# -----------------
#
# In your notebook, save the result with
# ``spectrum.write("my-processed-spectrum.scp")``.
# Use a new name for each result you want to keep. See the
# :doc:`first recipe </getting_started/quickstart>` for saving a figure
# and opening a saved spectrum.
#
# Continue with :doc:`analysis` to find and fit peaks, or
# :doc:`processing_2d` for a 2D acquisition.
