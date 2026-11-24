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
# Process an STMAS spectrum
# =========================
#
# Display the public rubidium-87 STMAS acquisition as a 2D map in ppm.
# The steps are the same as for :doc:`mqmas`: read, correct the common phase,
# straighten the pattern and plot a real plane.
#
# This recipe uses RbNO3 experiment 15 with its acquisition and processing
# parameter files. The read downloads the complete example folder if needed;
# keep an internet connection for that first run.
#
# Read, process and display
# -------------------------

# %%
import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/RbNO3/15/ser"
raw = scp.nmr.read(path, use_tdeff=True, indirect_dwell="IN[0]")
xf2 = scp.nmr.Experiment(raw).xf2(size=1024)
phased = scp.nmr.Experiment(xf2).apk2d(order=0)
spectrum = scp.nmr.Experiment(phased).xfshear(
    size=512, experiment="STMAS", f1_shift=-0.1
)
plane = scp.nmr.Experiment(spectrum).RR
_ = plane[:, -10.0:-60.0].plot(method="contour", start=0.1, nlevels=8)

# %% [raw] raw_mimetype="text/restructuredtext"
# You should see the resonances on a direct F2 axis and an isotropic F1 axis.
# The contour levels are relative to the strongest signal. Lower ``start``
# to show weaker features; raise it to hide more noise.
#
# Check the common phase
# ----------------------
#
# Display the first F2 row after correction. The same phase is applied to
# all rows; the automatic step corrects F2 only. Check the line shape before
# measuring peak areas. If this row is too weak in your own acquisition,
# choose a stronger row with ``apk2d(reference_row=..., order=0)``.

# %%
_ = scp.nmr.Experiment(phased).rsr(1).real.plot(xlim=(-60.0, -10.0))

# %% [raw] raw_mimetype="text/restructuredtext"
# Settings to adapt
# -----------------
#
# * Replace the quoted path with your complete experiment's ``ser`` file.
#   Keep the acquisition and processing parameter files alongside the signal.
# * ``size`` sets the number of FFT points. Increasing it does not create
#   experimental resolution that was not acquired.
# * ``f1_shift`` moves the result within the isotropic F1 window, in ppm;
#   use a shift chosen for your acquisition.
# * The bounds in ``plane[...]`` select the displayed region, in ppm.
#   Keep the complete ``spectrum`` for further work.

# %% [raw] raw_mimetype="text/restructuredtext"
# ``use_tdeff=True`` and ``indirect_dwell="IN[0]"`` belong to this archived
# acquisition. They select its complete pairs and its recorded time step.
# Use the settings established for your own acquisition.

# %% [raw] raw_mimetype="text/restructuredtext"
# Save and continue
# -----------------
#
# Save the complete result in your notebook with
# ``spectrum.write("processed-stmas.scp")``. Choose a new name to
# keep an earlier result. See :doc:`analysis` for peak integration and fitting.
#
# If a downloaded example reports missing parameter files, see
# :doc:`refreshing an incomplete example </getting_started/example_data>`.
#
# Data: Christian Fernandez, LCS, distributed through spectrochempy_data,
# CC BY 4.0. The :doc:`detailed method <stmas_details>` contains
# acquisition conventions, processing variants and reference comparisons.
