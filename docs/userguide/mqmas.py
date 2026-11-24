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
# Process an MQMAS spectrum
# =========================
#
# Turn the public aluminium-27 PST-5 acquisition into a 2D map. The recipe
# reads the signal, estimates a common phase, straightens the pattern and
# plots the real-real plane. The final axes are in ppm.
#
# Run the cells in the notebook opened during
# :doc:`installation </getting_started/installation>`. Keep the complete
# experiment folder, including its parameter files, when using your own data.
#
# Read, process and display
# -------------------------

# %%
import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/MQMAS_27AL_PST5/51/ser"
raw = scp.nmr.read(path)
xf2 = scp.nmr.Experiment(raw).xf2(size=4096, direct_points=2048)
phased = scp.nmr.Experiment(xf2).apk2d(order=0)
spectrum = scp.nmr.Experiment(phased).xfshear(
    size=512, experiment="3QMAS", f1_shift=-20.0
)
plane = scp.nmr.Experiment(spectrum).RR
_ = plane[52.0:25.0, 65.0:20.0].plot(method="contour", start=0.09, nlevels=8)

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
_ = scp.nmr.Experiment(phased).rsr(1).real.plot(xlim=(20.0, 65.0))

# %% [raw] raw_mimetype="text/restructuredtext"
# Settings to adapt
# -----------------
#
# * Replace the quoted path with your complete experiment's ``ser`` file.
# * ``size`` sets the number of FFT points. Increasing it does not create
#   experimental resolution that was not acquired.
# * ``f1_shift`` moves the result within the isotropic F1 window, in ppm;
#   use a shift chosen for your acquisition.
# * The bounds in ``plane[...]`` select the displayed region, in ppm.
#   Keep the complete ``spectrum`` for further work.

# %% [raw] raw_mimetype="text/restructuredtext"
# ``direct_points=2048`` retains that many acquired samples for this example.
# The phase is estimated before the map is straightened; the first-row check
# also helps spot a sign or line-shape problem.

# %% [raw] raw_mimetype="text/restructuredtext"
# Save and continue
# -----------------
#
# Save the complete result in your notebook with
# ``spectrum.write("processed-mqmas.scp")``. Choose a new name to
# keep an earlier result. See :doc:`analysis` for peak integration and fitting.
#
# Data: Christian Fernandez, LCS, distributed through spectrochempy_data,
# CC BY 4.0. The :doc:`detailed method <mqmas_details>` contains
# acquisition conventions, processing variants and reference comparisons.
