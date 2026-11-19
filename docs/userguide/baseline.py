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
# The :doc:`short SNIP gallery example </examples/gallery/analysis/plot_nmr_snip>`
# reproduces this comparison from the original FID.
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
# Use known peak-free intervals when they are available. The next example
# adds a plane with slopes along both axes to a separable Gaussian peak.
# F1 and F2 are calibrated frequency coordinates in Hz. The frequency-domain
# flags describe this constructed spectrum; a reader supplies them for
# imported spectra.

# %%
f1 = scp.Coord.linspace(-500, 500, 101, units="Hz", title="F1")
f2 = scp.Coord.linspace(-1000, 1000, 201, units="Hz", title="F2")
indirect = scp.gaussian(f1, pos=0, width=100, normalized=False)
direct = scp.gaussian(f2, pos=0, width=120, normalized=False)
peak = indirect.data[:, None] * direct.data[None, :]
background = 0.2 + 0.0001 * f2.data[None, :] + 0.0002 * f1.data[:, None]
plane = scp.NDDataset(peak + background, coordset=[f1, f2])
plane.meta.isfreq = [True, True]

along_f2 = scp.nmr.Experiment(plane).abs2(order=1, ranges=[(-1000, -600), (600, 1000)])
corrected_plane = scp.nmr.Experiment(along_f2).abs1(
    order=1, ranges=[(-500, -300), (300, 500)]
)
_ = scp.multiplot(
    [plane, corrected_plane],
    method="image",
    labels=["Before", "After ABS2 + ABS1"],
    nrow=1,
    ncol=2,
    vmin=0,
    vmax=1.4,
    figsize=(9, 4),
)

# %% [raw] raw_mimetype="text/restructuredtext"
# Both panels share the same intensity scale. Each command fits a degree-one
# polynomial independently to every trace in the specified direction.
# ``ranges`` selects fit points in Hz; the resulting polynomial is subtracted
# over the whole axis. Use ``limits`` as well if only an interval should be
# corrected. The source spectrum remains available for comparison.
#
# Here the Gaussian tails in the fit intervals are negligible, so the
# corrected map recovers the known peak. On measured data, avoid intervals
# containing broad peaks or dispersive tails: a good polynomial fit does not
# establish that the selected points represent baseline. Inspect both the
# map and individual slices before integrating. The second correction is
# useful when residual baseline remains along F1; it is not a mandatory
# processing step. The :doc:`gallery example
# </examples/gallery/analysis/plot_nmr_baseline_2d>` gives the short recipe.
#
# See the
# :external+core:doc:`SpectroChemPy baseline guide <userguide/processing/baseline>`
# for more methods and examples, including polynomial and asymmetric
# least-squares correction.
