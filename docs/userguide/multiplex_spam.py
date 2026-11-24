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
# Process a Multiplex SPAM spectrum
# =================================
#
# Select the 3Q signal from the public rubidium-87 Multiplex acquisition,
# then process it into a 2D map. After the selection, use the same processing
# steps as for :doc:`mqmas`.
#
# This recipe uses RbNO3 experiment 206, recorded on the same sample and
# at the same magnetic field as the :doc:`stmas` example.
# The read downloads its complete experiment folder if needed.
# Keep an internet connection for that first run.
#
# Read, process and display
# -------------------------

# %%
import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/RbNO3/206/ser"
raw = scp.nmr.read(path, phase_cycle="paired")
states = scp.nmr.Experiment(raw).select_coherence(3)
xf2 = scp.nmr.Experiment(states).xf2(size=4096)
phased = scp.nmr.Experiment(xf2).apk2d(order=0)
spectrum = scp.nmr.Experiment(phased).xfshear(size=512, f1_shift=7.0)
plane = scp.nmr.Experiment(spectrum).RR
_ = plane[:, -10.0:-60.0].plot(method="contour", start=0.1, nlevels=8)

# %% [raw] raw_mimetype="text/restructuredtext"
# You should see two main groups of resonances, as in the STMAS example.
# ``f1_shift=7.0`` keeps both groups together within the F1 window.
# Keep this setting when running this example.
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
# * ``select_coherence(3)`` selects 3Q. The available orders depend on the
#   recorded phase cycle; see the detailed method before choosing 5Q.
# * ``size`` sets the number of FFT points. It does not create experimental
#   resolution that was not acquired.
# * ``f1_shift`` selects the F1 window. Its value depends on your acquisition;
#   see the detailed method before changing it.
# * The bounds in ``plane[...]`` select the displayed region, in ppm.

# %% [raw] raw_mimetype="text/restructuredtext"
# Save and continue
# -----------------
#
# Save the complete result in your notebook with
# ``spectrum.write("processed-multiplex_spam.scp")``. Choose a new name to
# keep an earlier result. See :doc:`analysis` for peak integration and fitting.
#
# If a downloaded example reports missing parameter files, see
# :doc:`refreshing an incomplete example </getting_started/example_data>`.
#
# Data: Christian Fernandez, LCS, distributed through spectrochempy_data,
# CC BY 4.0. The :doc:`detailed method <multiplex_spam_details>` contains
# acquisition conventions, processing variants and reference comparisons.
