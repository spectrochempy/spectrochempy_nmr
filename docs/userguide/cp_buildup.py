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
# Cross-polarization buildup series
# =================================
#
# A contact-time series follows the intensity transferred during cross
# polarization (CP). Here fifteen separate carbon-13 acquisitions are arranged
# by contact time, processed with one common phase and integrated in three
# fixed spectral windows. A buildup–decay model describes the integrated
# response of a band. Its parameters are effective time scales for this
# experiment, rather than assignments of individual chemical species.
#
# The data are the CC BY 4.0 LCS series in
# ``nmrdata/bruker/tests/nmr/CP``, distributed through ``spectrochempy_data``.
# The :doc:`short gallery recipe </examples/gallery/analysis/plot_processing_cp_nmr>`
# keeps one region and one fit. This guide explains the choices and compares
# two models without changing the measured areas.
#
# Read and order the acquisitions
# -------------------------------
#
# Experiments 80–94 each contain one FID. The contact pulse duration is P15,
# reported by the reader with units. Read those values instead of assuming
# that experiment number is proportional to contact time. Convert them to
# milliseconds, sort the acquisitions together with their times, and preserve
# that order throughout processing.

# %%
import spectrochempy as scp

root = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/CP"
fids = [scp.nmr.read(root / str(n) / "fid") for n in range(80, 95)]
fids.sort(key=lambda fid: fid.meta.p[0][15].to("ms").magnitude)
contact_time = scp.Coord(
    [fid.meta.p[0][15].to("ms").magnitude for fid in fids],
    units="ms",
    title="CP contact time",
)
print(contact_time)

# %% [raw] raw_mimetype="text/restructuredtext"
# The contact-time coordinate is a series axis, not an indirect Fourier
# dimension. Transform only the direct FID. Before combining another series,
# check that all rows share the nucleus, acquired grid, receiver scaling and
# acquisition conditions, including recycle delay and number of scans. Changing
# these can also change the measured intensity.
#
# Process with one phase and one intensity scale
# ----------------------------------------------
#
# Apply the same exponential window, FFT size and phase to every FID. The
# 50 Hz window is defined on the acquired samples; zero filling to 8192 points
# gives a denser frequency grid. Neither operation creates extra acquired
# information.
#
# For this series, the phase angles below were estimated in TopSpin 4.5.0 on
# experiment 86 (1.4 ms), where all three regions are visible. They are in
# degrees at the first frequency bin. Keep them fixed for the entire series.
# The explicit ``digital_filter="pknl"`` correction reads the fractional group
# delay and applies it on the final FFT grid. Reading above retains the FID;
# do not additionally remove that delay at import. See :doc:`processing_1d`
# for the filter and phase conventions.

# %%
phased = [
    scp.nmr.Experiment(fid).process(
        apodization="em",
        lb=50.0,
        size=8192,
        digital_filter="pknl",
        phase="manual",
        phc0=37.365900,
        phc1=-205.449468,
    )
    for fid in fids
]
spectra = scp.NDDataset(
    [spectrum.real.data for spectrum in phased],
    coordset=[contact_time, phased[0].x],
    units=phased[0].units,
    title="signal",
)
_ = spectra.plot(method="lines", xlim=(205.0, 35.0), colorbar=True)

# %% [raw] raw_mimetype="text/restructuredtext"
# The plot uses real absorption spectra on their original common intensity
# scale. Check weak and strong rows before integration. Independently phasing
# each low-intensity row can turn noise or a phase-estimation error into an
# apparent change in buildup. Normalizing every row to its own maximum would
# remove the intensity information being measured.
#
# Integrate fixed chemical-shift windows
# --------------------------------------
#
# Use the same ppm limits for every contact time. The three windows below
# contain the bands of this acquisition. Simpson integration reduces only the
# spectral ``x`` dimension; contact time remains the independent coordinate.
# The code retains the measured baseline.

# %%
windows = [(165.0, 183.0), (91.0, 107.0), (62.0, 79.0)]
curves = [spectra[:, low:high].simpson(dim="x") for low, high in windows]
_ = scp.plot_multiple(
    curves,
    method="scatter",
    labels=[f"{low:g}–{high:g} ppm" for low, high in windows],
    title="Carbon-13 CP buildup",
    legend="upper right",
)

# %% [raw] raw_mimetype="text/restructuredtext"
# Inspect the limits for overlap and peak tails. A baseline offset contributes
# an area proportional to the integration width. If correction is required,
# choose it explicitly, apply a consistent rule to every row and repeat the
# integration; see :doc:`baseline`. Check how much the fitted constants change
# when the integration limits or baseline choice change. The true baseline and
# true band areas of this experimental series are not independently known.
#
# Fit a single buildup–decay law
# ------------------------------
#
# For the strongest band (62–79 ppm), use the empirical model
#
# .. math::
#
#    I(t)=A\left(1-\exp(-t/T_b)\right)\exp(-t/T_d).
#
# ``buildup`` and ``decay`` therefore have the same units as the fitted axis:
# milliseconds here. The positive amplitude has the units of the integrated
# signal. ``A`` is a scale parameter, not the maximum of this curve. The model
# assumes a rising transfer contribution multiplied by a decaying contribution;
# inspect its residuals before interpreting either time scale.


# %%
def cp_buildup(t, amplitude, buildup, decay):
    return amplitude * (1 - scp.exp(-t / buildup)) * scp.exp(-t / decay)


signal = curves[-1]
fitter = scp.Optimize(method="leastsq", usermodels={"cp_buildup": cp_buildup})
fitter.script = """
MODEL: CP
shape: cp_buildup
  $ amplitude: 75.0, 0.0, none
  $ buildup: 1.0, 0.01, none
  $ decay: 10.0, 0.01, none
"""
_ = fitter.fit(signal)
print(fitter.script)
prediction = fitter.predict()
ax = signal.plot_scatter(marker="o", label="Measured area", ylabel="Integrated area")
_ = prediction.plot(ax=ax, clear=False, label="One CP law")
_ = ax.legend()
_ = (signal - prediction).plot(marker="o", title="CP fit residuals")

# %% [raw] raw_mimetype="text/restructuredtext"
# The printed script contains the fitted values. The fit line connects values
# at the measured contact times. For this dataset and these settings, the
# one-law fit gives about 0.213 ms for buildup and 9.19 ms for decay. Residuals are measured minus fitted areas on
# the same time coordinate. Look for a systematic early-time or late-time
# pattern, rather than judging the fit only by its overlap with the data.
#
# Compare two buildup contributions
# ----------------------------------
#
# A second model sums two laws with independent amplitudes and time scales.
# Use exactly the same input areas. Extra parameters can reduce the residual
# without identifying two distinct physical populations.

# %%
two_laws = scp.Optimize(method="leastsq", usermodels={"cp_buildup": cp_buildup})
two_laws.script = """
MODEL: FAST
shape: cp_buildup
  $ amplitude: 40.0, 0.0, none
  $ buildup: 0.1, 0.01, none
  $ decay: 6.0, 0.01, none
MODEL: SLOW
shape: cp_buildup
  $ amplitude: 40.0, 0.0, none
  $ buildup: 1.0, 0.01, none
  $ decay: 7.0, 0.01, none
"""
_ = two_laws.fit(signal)
print(two_laws.script)
ax = signal.plot_scatter(marker="o", label="Measured area", ylabel="Integrated area")
_ = prediction.plot(ax=ax, clear=False, label="One CP law", ls="--")
_ = two_laws.predict().plot(ax=ax, clear=False, label="Two CP laws")
_ = ax.legend()
_ = (signal - two_laws.predict()).plot(marker="o", title="Two-law fit residuals")

# %% [raw] raw_mimetype="text/restructuredtext"
# A fitted buildup shorter than the first acquired contact time is constrained
# by extrapolation of the assumed law. Here the fast buildup is about
# 0.074 ms, below the first measured contact time of 0.1 ms. Check the sampled time range, parameter
# bounds and sensitivity to starting values. A fitted constant near a bound or
# a large change between starting values calls for more data or a simpler
# model. The labels FAST and SLOW describe the fitted terms; they do not assign
# resonances or chemical species.
#
# Keep the fitted script, measured areas and residuals together with the
# processing settings and window limits. A common phase makes the series
# consistent, but does not by itself establish quantitative CP populations.
# For a delay-dependent saturation-recovery series, see :doc:`relaxation_t1`;
# its independent variable and physical model are different.
