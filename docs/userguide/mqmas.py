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
# Multiple-quantum magic-angle spinning (MQMAS) is a 2D solid-state NMR
# method for half-integer quadrupolar nuclei such as ²⁷Al. Magic-angle
# spinning leaves residual second-order quadrupolar broadening. MQMAS
# correlates the evolution of a multiple-quantum coherence in F1 with
# the detected central transition in F2. Their anisotropic frequency
# shifts are proportional, so a shear removes this contribution from F1.
# The resulting isotropic dimension helps resolve sites with overlapping
# F2 lines. Their F1 positions retain both the isotropic chemical shift
# and an isotropic second-order quadrupolar shift.
#
# This tutorial uses a Bruker States acquisition of PST-5 to show the
# tilted line shapes after two Fourier transforms, the effect of the
# shear, and how apodization and baseline correction change the result.
# %% [raw] raw_mimetype="text/restructuredtext"
# See :cite:t:`frydman1995mqmas` for the original MQMAS experiment
# and :cite:t:`fernandez2011quadrupolar` for experimental strategies
# and applications.

# %% [markdown]
# ## Transform F2 and inspect the first row
#
# Read the Bruker SER with [read()](../reference/generated/spectrochempy_nmr.read.rst),
# then apply a Fourier transform along the direct
# acquisition axis F2. TopSpin calls this step XF2. The indirect axis F1
# stays in the time domain so you can inspect and phase an F2 row before
# shearing.
# For PST-5, keep 2048 acquired complex samples in F2 and zero-fill to
# 4096 for the F2 transform. This samples the same spectrum on a finer grid before
# phase adjustment and quadrature reconstruction. The acquisition interval
# and the physical resolution are set by the retained samples.

# %%
import numpy as np
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
# preview. Select the first row with [rsr()](../reference/generated/spectrochempy_nmr.Experiment.rsr.rst)
# and estimate its phase with [apk()](../reference/generated/spectrochempy_nmr.autophase.rst),
# the alias for [autophase()](../reference/generated/spectrochempy_nmr.autophase.rst).
# Inspect the preview before shearing.
# The phase angles are in degrees and the pivot is an F2 sample index.
# The sign convention matches SpectroChemPy's [pk()](https://www.spectrochempy.fr/latest/reference/generated/spectrochempy.pk.html). To reproduce
# TopSpin phase values with a pivot of zero, reverse the signs of PHC0
# and PHC1. Keep the original complex XF2 dataset for the next step.
#
# ## Inspect the spectrum before shearing
#
# Apply the selected phase with [xf2p()](../reference/generated/spectrochempy_nmr.Experiment.xf2p.rst)
# to every F1 increment and both States branches. Combine the two real
# F2 branches as the real and imaginary parts of the indirect signal.
# Transform F1 directly to see why shearing is needed. The dashed line
# shows the predicted direction of the quadrupolar broadening for one
# site. Each site has its own intercept. The F1 DFT is periodic, so the
# displayed frequency offsets use the period that contains both peaks.

# %%
phased = scp.nmr.Experiment(xf2).xf2p(**phase)
f1_size = 512
shear = 19 / 12
# The F2 grid is in descending ppm. Convert to signed Hz from its center.
f2_carrier = phased.x.data[phased.shape[-1] // 2]
reference_mhz = float(raw.meta.sfo1[-1].to("MHz").magnitude) / (1 + f2_carrier * 1e-6)
f2_hz = (f2_carrier - phased.x.data) * reference_mhz
dwell = phased.y.data[1] - phased.y.data[0]
f1_hz = np.arange(-f1_size, 0) / (f1_size * dwell)
states = phased.data[0].real + 1j * phased.data[1].real
states[0] *= 0.5
# Use the equivalent negative-frequency period to keep both sites in view.
unsheared = f1_size * np.fft.ifft(states, n=f1_size, axis=0)
unsheared_rr = scp.NDDataset(
    unsheared.real,
    units=phased.units,
    coordset=[
        scp.Coord(f1_hz, units="Hz", title="F1 offset"),
        scp.Coord(f2_hz, units="Hz", title="F2 offset"),
    ],
)
# %% [markdown]
# ## Shear and transform F1
#
# For the positive-exponent Fourier convention used here, multiply the
# complex States signal at each direct frequency $f_2$ by a phase ramp
# in the indirect evolution time $t_1$.
#
# $$s_{\mathrm{shear}}(t_1,f_2) = s(t_1,f_2)\,e^{2\pi i k f_2 t_1},
# \qquad f_{1,\mathrm{shear}} = f_1 - k f_2.$$
#
# The slope is $k=19/12$ for this 3Q experiment on ²⁷Al. If the
# broadening at a site follows $f_1 = k f_2 + b$, the shear puts
# it at a constant $f_{1,\mathrm{shear}}=b$. The anisotropic line
# shape remains visible along F2. The plugin method
# [xfshear()](../reference/generated/spectrochempy_nmr.Experiment.xfshear.rst)
# combines the shear and F1 transform, then calibrates F1 in isotropic
# ppm, with a frequency scale divisor of $3-k=17/12$.

# %%
spectrum = scp.nmr.Experiment(phased).xfshear(
    size=f1_size, spin=2.5, coherence_order=3, f1_shift=-20.0
)
rr = scp.nmr.Experiment(spectrum).rr()
region = rr[52.0:25.0, 65.0:20.0]
peak = region.max().magnitude
comparison_levels = [
    peak * fraction for fraction in (0.06, 0.1, 0.16, 0.25, 0.4, 0.65, 0.9)
]
f2_bounds = tuple((f2_carrier - ppm) * reference_mhz for ppm in (65, 20))
unsheared_region = unsheared_rr[-10500.0:-4200.0, f2_bounds[0] : f2_bounds[1]]
strongest = np.unravel_index(unsheared_region.data.argmax(), unsheared_region.shape)
f2_peak = unsheared_region.x.data[strongest[1]]
f1_peak = unsheared_region.y.data[strongest[0]]
axes = scp.multiplot(
    [unsheared_region, region],
    ncol=2,
    method="contour",
    labels=["Two Fourier transforms", "After shearing"],
    levels=comparison_levels,
    cmap="magma_r",
    equal_aspect=False,
    figsize=(10, 4),
)
line_x = np.array(f2_bounds)
_ = axes[0].plot(line_x, f1_peak + shear * (line_x - f2_peak), "--", color="0.3")

# %% [markdown]
# To revise the F2 phase, return to the complex XF2 dataset and run
# [xf2p()](../reference/generated/spectrochempy_nmr.Experiment.xf2p.rst) again with the new angles. Use [pk()](https://www.spectrochempy.fr/latest/reference/generated/spectrochempy.pk.html) on the preview row
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
# of the resonances with [abs2()](../reference/generated/spectrochempy_nmr.Experiment.abs2.rst).
# Extract each real-real plane with [rr()](../reference/generated/spectrochempy_nmr.Experiment.rr.rst)
# and display them on common contour levels with [multiplot()](https://www.spectrochempy.fr/latest/reference/generated/spectrochempy.multiplot.html).

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
peak = region.max().magnitude
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


# %% [markdown]
# ## Compare exponential echo windows
#
# Use the public PST-5 acquisition to compare the unweighted signal
# with a symmetric 40 Hz exponential echo and antiecho window.
# The contour levels are shared across the two panels.

# %%
xf2_symmetric = scp.nmr.Experiment(raw).xf2(
    size=4096,
    direct_points=2048,
    mqecho=(19 / 12, 40),
)
symmetric_phased = scp.nmr.Experiment(xf2_symmetric).xf2p(**phase)
symmetric = scp.nmr.Experiment(symmetric_phased).xfshear(
    size=512,
    spin=2.5,
    f1_shift=-20.0,
)
symmetric_rr = scp.nmr.Experiment(symmetric).rr()[52.0:25.0, 65.0:20.0]
_ = scp.multiplot(
    [region, symmetric_rr],
    ncol=2,
    method="contour",
    labels=["No window", "Symmetric echo window (40 Hz)"],
    levels=comparison_levels,
    cmap="magma_r",
    figsize=(10, 4),
)
