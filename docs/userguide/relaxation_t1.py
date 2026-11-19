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
# Saturation-recovery T1 series
# =============================
#
# A saturation-recovery series measures signal as a function of recovery delay.
# Read the delay list as a physical coordinate, process only the direct FID
# axis, integrate a fixed resonance window and fit a recovery model. The delay
# axis remains in time units throughout: it is not an F1 Fourier dimension.
#
# The worked example uses ``nmrdata/bruker/tests/nmr/relax/100``, a silicon
# CP/MAS saturation-recovery acquisition distributed as LCS data through
# ``spectrochempy_data`` under CC BY 4.0. The :doc:`short gallery recipe
# </examples/gallery/analysis/plot_processing_nmr_relax>` runs the same workflow.
# The comparison here concerns these measured signals and the stated model;
# it does not reproduce the vendor's fitting tools.
#
# Read the FIDs and recovery delays
# ---------------------------------
#
# This acquisition stores the series in one ``ser`` file and its recovery
# delays in ``vdlist``. Pass the list name explicitly. The reader supplies the
# list values as the secondary coordinate, paired with the stored rows. Do not
# replace nonuniform delays by row indices or an evenly spaced axis.

# %%
import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/relax/100/ser"
raw = scp.nmr.read(path, use_list="vdlist", remove_digital_filter=True)
print(raw.y)

# %% [raw] raw_mimetype="text/restructuredtext"
# Check the delay values and their units before fitting another acquisition.
# Each row must correspond to one list entry. These data use seconds, so the
# fitted T1 below is in seconds. Confirm from the acquisition that the list
# represents recovery delays, rather than another pulse-sequence parameter.
# Keep acquisition conditions and intensity scaling comparable across rows.
#
# Window, transform and phase the direct axis
# -------------------------------------------
#
# Use the same 15 Hz exponential window and phase for every row. In this
# example the reader already corrects the digital-filter delay in the time
# domain. Do not apply a second correction later. The following explicit
# operations transform ``x`` only and leave the delay coordinate unchanged.
#
# Estimate one zero-order phase from the longest-delay row, which has a
# strong signal in this acquisition. Use the −140 to −80 ppm band for the
# estimate, then apply that same phase to the complete spectral window at
# every delay. Inspect the full spectra before selecting the integration
# window. Phasing each weak row independently can bias the recovery curve.

# %%
weighted = raw.em(lb=15 * scp.ur.Hz, dim="x")
f2 = weighted.fft(dim="x")
phase = scp.nmr.apk(f2[-1].squeeze()[-140.0:-80.0], order=0, apply=False)
if not phase.estimate.success:
    raise RuntimeError(phase.estimate.message)
print(phase.estimate)
spectrum = f2.pk(phc0=phase.estimate.p0 * scp.ur.deg, phc1=0 * scp.ur.deg)
_ = spectrum.real.plot(
    method="lines",
    palette="viridis",
    colorbar=True,
    title="Saturation recovery — complete spectral window",
)

# %% [raw] raw_mimetype="text/restructuredtext"
# The full window is approximately +458 to −526 ppm. These spectra share one
# intensity scale; no row is normalized to its own maximum. The phase estimate
# for these data is about 104°. A predominantly dispersive line is
# unsuitable for integrating its real component; inspect the phase before
# selecting a signal for the recovery fit.
#
# Zoom into the resonance
# -----------------------
#
# Select −120 to −85 ppm around the silicon signal near −103 ppm, leaving
# room on the left of the line. Plot this region and retain the same data for
# integration below, without further phasing, baseline correction or
# normalization.

# %%
resonance = spectrum[:, -120.0:-85.0].real
_ = resonance.plot(
    method="lines",
    palette="viridis",
    colorbar=True,
    xlim=(-85.0, -120.0),
    title="Same phased spectra — expanded view",
)

# %% [raw] raw_mimetype="text/restructuredtext"
# Inspect both views for absorption, tails and baseline before integration.
# A successful phase estimator is a diagnostic, not proof of an ideal phase
# across the whole window. The imaginary component can also help diagnose
# residual phase errors. The recovery signal below uses the real component;
# magnitude would change the treatment of signed noise and weak signals.
#
# Integrate a consistent resonance window
# ---------------------------------------
#
# Integrate the same −120 to −85 ppm window displayed above for every delay.
# Simpson integration acts on the spectral dimension and returns one area
# per recovery delay.

# %%
signal = resonance.simpson(dim="x")
_ = signal.plot_scatter(marker="o", title="Integrated recovery signal")
print(signal)

# %% [raw] raw_mimetype="text/restructuredtext"
# Check that the window contains the whole resonance without unrelated
# signals. The code retains the measured baseline. If correction is needed,
# use a consistent method and signal-free regions across the series; inspect
# how the integrated areas change. See :doc:`baseline`. A varying baseline
# can masquerade as a change in recovery, and a fixed baseline offset can
# bias a model that assumes zero signal at zero delay.
#
# Fit a saturation-recovery model
# --------------------------------
#
# For ideal saturation followed by a single exponential recovery, use
#
# .. math::
#
#    I(t)=I_0\left(1-\exp(-t/T_1)\right).
#
# The model fixes ``I(0)=0`` and approaches ``I0`` at long delay. It has two
# positive fitted parameters: the limiting signal and T1. It is a
# saturation-recovery model; an inversion-recovery experiment needs its own
# initial condition and model. If saturation is incomplete or several
# components contribute, this two-parameter curve may be inadequate.
#
# The user model receives the numerical delay values. Because the coordinate
# is in seconds, both the starting value 2.0 and the lower bound 0.1 for ``T1``
# are in seconds. The amplitude bounds are expressed on the measured area
# scale. They are example settings to inspect for each acquisition.


# %%
def recovery(t, I0, T1):
    return I0 * (1 - scp.exp(-t / T1))


fitter = scp.Optimize(method="leastsq", usermodels={"recovery": recovery})
fitter.script = """
MODEL: T1
shape: recovery
  $ I0: 4000.0, 1.0, none
  $ T1: 2.0, 0.1, none
"""
_ = fitter.fit(signal)
print(fitter.script)
prediction = fitter.predict()
ax = signal.plot_scatter(marker="o", label="Measured area", ylabel="Integrated area")
_ = prediction.plot(ax=ax, clear=False, label="Saturation-recovery fit")
_ = ax.legend()
_ = (signal - prediction).plot(marker="o", title="Recovery fit residuals")

# %% [raw] raw_mimetype="text/restructuredtext"
# Read T1 from the printed fitted script in seconds. The estimate depends
# on the phase and integration settings above. The fitted line connects
# predictions at the measured delays. Residuals are measured minus fitted
# areas at those same, potentially nonuniform delays.
#
# Assess the fit before interpreting T1
# -------------------------------------
#
# Inspect early and late delays separately. A systematic residual can indicate
# an inappropriate initial condition, baseline error or a recovery that is not
# single exponential. Compare the longest delays with the fitted plateau;
# without a well-sampled plateau, ``I0`` and T1 can compensate for each other.
# A close visual fit alone does not establish parameter precision.
#
# Repeat with different initial guesses, check whether fitted values approach
# the imposed bounds, and vary reasonable integration limits and baseline
# choices. Replicate acquisitions and an appropriate noise model are needed
# for experimental uncertainty; this example does not turn the fitted curve
# into a confidence interval. CP detection also weights the observed signal,
# so interpret the fitted T1 as the recovery of this selected signal under
# the acquisition conditions.
#
# To analyze another resolved band, repeat the integration and fit with that
# band's fixed ppm limits. Do not reuse a result merely because the delay axis
# is shared. Preserve the delay coordinate and its units, phase settings,
# integration limits, fitted script and residuals with the result.
# See :doc:`cp_buildup` for a series whose independent variable is CP contact
# time rather than recovery delay.
