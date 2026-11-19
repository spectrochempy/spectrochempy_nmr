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
# Phase correction along F2
# =========================
#
# Phase complex data before extracting an absorptive real plane. For paired
# 2D acquisitions retain both branches throughout processing. This page
# covers automatic F2 phase estimation; it does not estimate indirect phase.
# See :doc:`processing_2d` for the overall workflow and :doc:`components_2d`
# for selecting the resulting plane.
#
# Common and individual row phases
# --------------------------------
#
# For a series of complex F2 spectra, ``apk2d()`` estimates a common phase
# from the first row by default. Pass ``reference_row`` to select another row
# using TopSpin's 1-based numbering. Use ``per_row=True`` when the phase varies
# across the series. Inspect individual estimates when the signal becomes
# weak, especially at high diffusion weightings. The estimator uses the same options as ``scp.nmr.apk()``.

# %%
import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/synthetic/simpson.fid"
reference = scp.nmr.Experiment(scp.nmr.read(path)).process()
shifted = reference.pk(phc0=35.0)
series = scp.NDDataset(
    [scale * shifted.data for scale in (1.0, 0.6, 0.3)],
    coordset=[scp.Coord([1, 2, 3], title="Trace"), shifted.x],
)
series.meta.isfreq = [False, True]
phased = scp.nmr.Experiment(series).apk2d(reference_row=1, order=0)
_ = scp.plot_multiple(
    [series[0].squeeze().real, phased[0].squeeze().real],
    labels=["Before", "After common F2 phase"],
    legend="best",
    xlim=(-800, -200),
)

# %% [raw] raw_mimetype="text/restructuredtext"
# For MQMAS, pass the complex result of ``xf2()``. The common phase applies
# to both States branches. The output can go directly to ``xfshear()``.
# An unsuccessful estimate raises an error identifying the row.
#
# For an executable before/after comparison with a known common phase error,
# see :doc:`/examples/gallery/analysis/plot_nmr_apk2d`. The secondary axis in
# that example indexes spectra, rather than a second Fourier dimension.
# A common reference-row phase preserves relative amplitudes across a series.
# Independent per-row estimates can become unstable on weak spectra; use
# them only when there is evidence that phase actually varies between rows.
# ``apk2d`` estimates F2 phase, not the phase of the indirect dimension.
# Generate the synthetic input once as described in :doc:`reader_workflows`.
#
