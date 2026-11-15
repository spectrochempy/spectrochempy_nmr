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
# Apodization in NMR
# ==================
#
# Apodization multiplies the complex FID by a time-domain window before
# zero filling and Fourier transformation. Both the real and imaginary
# samples receive the same weights. This changes the balance between
# noise, peak width and truncation artifacts in the spectrum.
#
# Choosing a window
# -----------------
#
# * **Exponential** (``em``). A positive ``lb`` in Hz gives less weight
#   to late samples. This can improve the signal-to-noise ratio for a
#   decaying FID, at the cost of broader Lorentzian lines. Increasing
#   ``lb`` strengthens both effects.
# * **Gaussian** (``gm``). Use ``gb`` in Hz to control the Gaussian
#   damping. With ``lb=0`` the window is purely Gaussian. A positive
#   ``lb`` adds inverse exponential weighting to compensate some of the
#   original decay. Inspect the window and the spectrum, since excessive
#   compensation can amplify noise.
# * **Sine bell** (``sp``). Use ``ssb=1`` for a window that starts near
#   zero, or ``ssb=2`` for a cosine-shaped window that starts at full
#   weight. Set ``pow=2`` to square the window. Sine-bell windows taper
#   the end of the acquisition and are often used along indirect
#   dimensions in 2D NMR.
#
# Compare windows on a Bruker FID
# -------------------------------
#
# Read the same FID as in :doc:`processing_1d`, with the digital-filter
# delay corrected. Request the window with ``retapod=True`` to inspect
# the weights alongside the weighted FID. The parameters below illustrate
# the three window shapes on this acquisition.

# %%
import spectrochempy as scp

fid = scp.nmr.read_topspin(
    "nmrdata/bruker/tests/nmr/topspin_1d/1/fid", remove_digital_filter=True
)
exponential, exponential_window = fid.em(lb=30.0, retapod=True)
gaussian, gaussian_window = fid.gm(gb=75.0, lb=0.0, retapod=True)
cosine_bell, cosine_window = fid.sp(ssb=2, pow=2, retapod=True)
labels = [
    "Unweighted",
    "Exponential (30 Hz)",
    "Gaussian (75 Hz)",
    "Squared cosine bell",
]
fids = [fid, exponential, gaussian, cosine_bell]
windows = [
    scp.ones_like(exponential_window),
    exponential_window,
    gaussian_window,
    cosine_window,
]

for dataset in [*fids, *windows]:
    dataset.x = dataset.x.to("ms")

_ = scp.plot_multiple(
    windows,
    method="pen",
    labels=labels,
    legend="lower left",
    xlabel="Acquisition time / ms",
    ylabel="Window weight",
    ylim=(-0.05, 1.05),
)
_ = scp.plot_multiple(
    [abs(dataset) for dataset in fids],
    method="pen",
    labels=labels,
    legend="upper right",
    xlabel="Acquisition time / ms",
    ylabel="FID magnitude / count",
)

# %% [raw] raw_mimetype="text/restructuredtext"
# Compare the spectra
# -------------------
#
# Use the same zero filling and phase correction for all four traces to
# isolate the effect of the window. Compare peak width, height and the
# surrounding baseline on the same intensity scale. The stronger exponential
# and Gaussian settings make the changes near the main resonance visible.

# %%
spectra = [
    dataset.zf_size(size=32768).fft().pk(phc0=52.44, phc1=-16.84) for dataset in fids
]
_ = scp.plot_multiple(
    [spectrum[2.0:-0.5] for spectrum in spectra],
    method="pen",
    labels=labels,
    legend="upper left",
    xlim=(2, -0.5),
)

# %% [raw] raw_mimetype="text/restructuredtext"
# The exponential and Gaussian windows reduce the height of the main
# peak in this example. The squared cosine bell leaves it close to the
# unweighted result because most of the signal has already decayed by
# the time that window tapers strongly.
#
# Use the same settings in a processing workflow
# ----------------------------------------------
#
# Set ``apodization="em"``, ``"gm"`` or ``"sp"`` in
# ``Experiment(fid).process()`` to include a window in the 1D workflow.
# For example, ``Experiment(fid).process(apodization="em", lb=30.0)``
# applies the exponential window before the FFT.
#
# When processing a 2D acquisition step by step, specify the time-domain
# dimension explicitly, for example ``fid.em(lb=5.0, dim="x")`` before
# the F2 transform. Choose the indirect window separately while F1 is
# still in the time domain. See :doc:`processing_2d` for the transform order.
