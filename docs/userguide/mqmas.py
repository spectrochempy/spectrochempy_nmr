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

# %% [markdown]
# # Processing MQMAS spectra
#
# > The staged MQMAS workflow is available in the development version of
# > the plugin.
#
# See the [changelog](../changelog.rst) for release status.
#
# The XFSHEAR workflow separates isotropic and second-order quadrupolar
# contributions in MQMAS spectra. It uses a Bruker States acquisition with a
# common F2 phase and isotropic F1 calibration.
#
# ## Transform F2 and inspect the first row
#
# Read the Bruker SER, then apply a Fourier transform along the direct
# acquisition axis F2. TopSpin calls this step XF2. The indirect axis F1
# stays in the time domain so you can inspect and phase an F2 row before
# shearing.
# For PST-5, keep 2048 acquired complex samples in F2 and zero-fill to
# 4096 for the F2 transform. This samples the same spectrum on a finer grid before
# phase adjustment and quadrature reconstruction. The acquisition interval
# and the physical resolution are set by the retained samples.

# %%
import spectrochempy as scp

raw = scp.nmr.read("nmrdata/bruker/tests/nmr/MQMAS_27AL_PST5/51/ser")
xf2 = scp.nmr.Experiment(raw).xf2(size=4096, direct_points=2048)
first_row = scp.nmr.Experiment(xf2).rsr(1)
preview = scp.nmr.apk(first_row[65.0:20.0], order=0)
if not preview.estimate.success:
    raise RuntimeError(preview.estimate.message)
phase = {"phc0": preview.estimate.p0, "phc1": preview.estimate.p1, "pivot": 0}
_ = preview.phased.real.plot()

# %% [markdown]
# By default, [xf2()](../reference/generated/spectrochempy_nmr.Experiment.xf2.rst) corrects the digital-filter delay before the phase
# preview. The `apk()` alias for `autophase()` estimates the phase on
# the first row alone. Inspect the preview before running XFSHEAR.
# The phase angles are in degrees and the pivot is an F2 sample index.
# The sign convention matches SpectroChemPy's `pk()`. To reproduce
# TopSpin phase values with a pivot of zero, reverse the signs of PHC0
# and PHC1. Keep the original complex XF2 dataset for the next step.
#
# ## Shear and transform F1
#
# Apply the selected phase with [xf2p()](../reference/generated/spectrochempy_nmr.Experiment.xf2p.rst) to every F1 increment and both
# States branches. Pass the phased dataset and the F1 FFT size to [xfshear()](../reference/generated/spectrochempy_nmr.Experiment.xfshear.rst).
# The shear includes the F1 transform.

# %%
phased = scp.nmr.Experiment(xf2).xf2p(**phase)
spectrum = scp.nmr.Experiment(phased).xfshear(
    size=512, spin=2.5, coherence_order=3, f1_shift=-20.0
)
rr = scp.nmr.Experiment(spectrum).rr()
_ = rr[52.0:25.0, 65.0:20.0].plot(method="contour")

# %% [markdown]
# To revise the F2 phase, return to the complex XF2 dataset and run
# [xf2p()](../reference/generated/spectrochempy_nmr.Experiment.xf2p.rst) again with the new angles. Use `pk()` on the preview row
# to select the angles and [xf2p()](../reference/generated/spectrochempy_nmr.Experiment.xf2p.rst) on the complete dataset to record them
# for XFSHEAR. Use `f1_shift` to move the signal within the F1 window, in isotropic ppm.
# For shifted-echo acquisitions, `echo_window=(slope, sigma)` in
# [xf2()](../reference/generated/spectrochempy_nmr.Experiment.xf2.rst) applies a Gaussian along F2 before its FFT.
# Its center follows the expected echo at each F1 increment and includes
# the digital group delay. Choose the slope for the acquisition and give
# `sigma` in seconds. For baseline correction, fit the selected real plane
# along F2 using signal-free regions.
#
# Use `mqecho=(slope, lb_hz)` in [xf2()](../reference/generated/spectrochempy_nmr.Experiment.xf2.rst) or [prepare_xfshear()](../reference/generated/spectrochempy_nmr.Experiment.prepare_xfshear.rst) to
# average two exponential windows centered at opposite slopes in the
# group-delay-corrected time grid. For PST-5, use
# `xf2(size=4096, direct_points=2048, mqecho=(19 / 12, 40))`.
# Supply either window option. Both States branches receive the same
# weights on the raw sample grid, which begins at the negative group
# delay. When the mirrored center lies outside that grid, only its tail
# overlaps the samples. See the [exponential-window comparison](stmas_multiplex.rst).
#
# The returned channels pack RR and IR in `S_R`, and RI and II in
# `S_I`. The F2 axis retains the acquisition reference. The F1 axis
# uses the isotropic scaling and any requested F1 shift.
#
# The [PST-5 gallery notebook](../examples/gallery/nmr/plot_replay_mqmas_pst5.rst)
# shows the workflow from the raw SER to the RR map.
# See the [processing conventions](../devguide/processing_conventions.rst)
# for the TopSpin comparisons.
#
# For the MQMAS conventions and broader context, see
# [Fernandez and Pruski (2011)](https://doi.org/10.1007/128_2011_141).
#
# ## Estimate and apply the common phase in one call
#
# Use [apk2d()](../reference/generated/spectrochempy_nmr.Experiment.apk2d.rst) on the complex XF2 dataset when you want to estimate the
# phase from one row and apply it to every increment and both States branches.
# The following call uses the first row and the same 20–65 ppm window as the
# preview above.

# %%
exclude = (first_row.x.data < 20.0) | (first_row.x.data > 65.0)
phased = scp.nmr.Experiment(xf2).apk2d(reference_row=1, order=0, exclude=exclude)
spectrum = scp.nmr.Experiment(phased).xfshear(
    size=512, spin=2.5, coherence_order=3, f1_shift=-20.0
)

# %% [markdown]
# ## Compare apodization and baseline correction
#
# Keep the original phase and shear settings to isolate the effect of the
# Gaussian window. On PST-5 the echo moves by (19/12) × t1 along F2, in
# addition to the digital group delay. Apply the same window to both States
# branches at a given increment, with a Gaussian standard deviation of
# 0.8 ms. Then fit a linear baseline along F2 using regions to either side
# of the resonances.

# %%
xf2_windowed = scp.nmr.Experiment(raw).xf2(
    size=4096, direct_points=2048, echo_window=(19 / 12, 0.0008)
)
phased_windowed = scp.nmr.Experiment(xf2_windowed).xf2p(**phase)
windowed = scp.nmr.Experiment(phased_windowed).xfshear(
    size=512, spin=2.5, f1_shift=-20.0
)
windowed_rr = scp.nmr.Experiment(windowed).rr()[52.0:25.0, 65.0:20.0]

corrected_spectrum = scp.nmr.Experiment(windowed).abs2(
    order=1, limits=(20.0, 65.0), ranges=[(21.0, 24.0), (61.0, 64.0)]
)
corrected = scp.nmr.Experiment(corrected_spectrum).rr()[52.0:25.0, 65.0:20.0]

region = rr[52.0:25.0, 65.0:20.0]
peak = float(scp.Quantity(region.max()).magnitude)
positive_levels = [0.08 * peak * 1.3**i for i in range(10)]
levels = [-level for level in reversed(positive_levels)] + positive_levels
_ = scp.multiplot(
    [region, windowed_rr, corrected],
    ncol=3,
    method="contour",
    labels=["Reference", "F2 apodization", "F2 apodization and baseline correction"],
    levels=levels,
    figsize=(13, 4),
)

# %% [markdown]
# Inspect weak negative features with a filled contour map whose color
# limits are symmetric about zero. The 99th percentile of the absolute
# intensities sets the limits.

# %%
_ = corrected.plot(
    method="contourf", symmetric_percentile=99, cmap="RdBu_r", colorbar=True
)
