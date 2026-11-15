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
# Baseline correction in NMR
# ==========================
#
# A sloping or curved baseline contributes to peak areas and can bias a
# fit. Work on the real component of a phased 1D spectrum or on the
# spectral plane selected for 2D analysis. Check the phase first,
# because a dispersive tail is part of a resonance and can distort a
# baseline estimate. See :doc:`processing_1d` for phase correction.
#
# Compare a spectrum before and after correction
# -----------------------------------------------
#
# Use the same phased Bruker spectrum as in :doc:`analysis`. SNIP
# estimates the baseline by iteratively clipping positive peaks.
# This example uses it on a spectrum dominated by positive resonances.
# Apply it to the full spectrum and then zoom in to compare the result
# and the subtracted baseline.

# %%
import spectrochempy as scp

fid = scp.nmr.read_topspin(
    "nmrdata/bruker/tests/nmr/topspin_1d/1/fid", remove_digital_filter=True
)
spectrum = scp.nmr.Experiment(fid).process(
    apodization="em", lb=2.0, size=32768, phase="manual", phc0=52.44, phc1=-16.84
)
real_spectrum = spectrum.real
corrected = real_spectrum.snip(snip_width=50)
baseline = real_spectrum - corrected

_ = scp.plot_multiple(
    [trace[9.0:-1.0] for trace in (real_spectrum, corrected, baseline)],
    method="pen",
    labels=["Before correction", "After SNIP", "Subtracted baseline"],
    legend="upper left",
)

# %% [raw] raw_mimetype="text/restructuredtext"
# Inspect the baseline through broad and overlapping signals before
# integrating. Adjust ``snip_width`` in samples and compare the estimated
# baselines if you change the point spacing or process a different
# acquisition. For signed peaks, a polynomial fitted to signal-free
# regions is another option. Choose those regions from the spectrum and
# check that the correction does not remove broad resonances.
#
# Correct a 2D spectrum
# ---------------------
#
# For a phased 2D spectrum with a baseline that varies by trace, the
# plugin's ``Experiment.abs2()`` fits a polynomial along each F2 row.
# Use ``Experiment.abs1()`` to correct along F1 columns. These TopSpin-style
# operations correct the RR plane. See :ref:`topspin-slice-and-baseline`
# for limits, fit regions, and a worked call.
#
# See the
# :external+core:doc:`SpectroChemPy baseline guide <userguide/processing/baseline>`
# for more methods and examples, including polynomial and asymmetric
# least-squares correction.
