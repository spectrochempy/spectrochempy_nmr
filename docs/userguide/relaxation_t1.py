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
# Measure a recovery series
# =========================
#
# Follow the signal as the recovery delay increases. This public silicon
# series contains nine spectra. Read their delays, apply the same processing
# to every row and integrate the same band at each delay.
#
# Read and display the spectra
# ----------------------------
#
# The ``vdlist`` file contains the recovery delays in seconds. Keep it with
# the ``ser`` file and the experiment's parameter files. The recipe estimates
# one phase from row 9, the longest-delay spectrum, then applies it to all rows.

# %%
import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/relax/100/ser"
raw = scp.nmr.read(path, use_list="vdlist", remove_digital_filter=True)
f2 = raw.em(lb=15 * scp.ur.Hz, dim="x").fft(dim="x")
band = f2[:, -140.0:-80.0]
phased = scp.nmr.Experiment(band).apk2d(reference_row=9, order=0)
region = phased[:, -120.0:-85.0].real
_ = region.plot(method="lines", colorbar=True, title="Recovery series")

# %% [raw] raw_mimetype="text/restructuredtext"
# You should see the band near -103 ppm grow with delay. Check the line shape
# and the baseline between peaks before integrating. Avoid scaling each
# spectrum to its own maximum: the change in signal is what we want to measure.
#
# Measure the same band at every delay
# ------------------------------------
#
# Integrate the -120 to -85 ppm region selected above. The resulting points
# retain the recorded delays, including their nonuniform spacing.

# %%
signal = region.simpson(dim="x")
_ = signal.plot_scatter(
    marker="o", xlabel="Recovery delay / s", ylabel="Integrated area / count ppm"
)

# %% [raw] raw_mimetype="text/restructuredtext"
# The points should rise towards a plateau. This is an area measurement,
# not a concentration or a fitted relaxation time.
#
# Settings to adapt
# -----------------
#
# * Replace the quoted path with your experiment's ``ser`` file, keeping its
#   delay list and parameter files together.
# * ``lb=15 * scp.ur.Hz`` sets the smoothing, in Hz.
# * ``reference_row=9`` chooses a strong row for the common phase in this
#   example. Use a suitable row for your data.
# * ``-140.0:-80.0`` is the band used to estimate phase; ``-120.0:-85.0`` is
#   the band integrated. Change both ranges for another signal, in ppm.
#
# Use the same phase and integration window at every delay. A changing
# baseline can also change the measured area; see :doc:`baseline` if needed.
#
# Save and fit the recovery
# -------------------------
#
# In your notebook, save the measured curve with
# ``signal.write("recovery-areas.scp")``. Choose a new name to keep an earlier
# result. Continue with :doc:`recovery_details` to estimate an apparent
# recovery time or fit overlapping bands across the series. The fitted time
# will depend on the chosen signal and model.
#
# For a contact-time experiment, see :doc:`cp_buildup`.
#
# Data: Christian Fernandez, LCS, distributed through spectrochempy_data,
# CC BY 4.0.
