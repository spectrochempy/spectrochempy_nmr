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
import matplotlib.pyplot as plt

import spectrochempy as scp

fid = scp.nmr.read_topspin(
    "nmrdata/bruker/tests/nmr/topspin_1d/1/fid", remove_digital_filter=True
)
comparisons = [
    ("Exponential (30 Hz)", *fid.em(lb=30.0, retapod=True)),
    ("Gaussian (75 Hz)", *fid.gm(gb=75.0, lb=0.0, retapod=True)),
    ("Squared cosine bell", *fid.sp(ssb=2, pow=2, retapod=True)),
]

fig, (weights_ax, fid_ax) = plt.subplots(2, 1, sharex=True, layout="constrained")
time = fid.x.to("ms").data
weights_ax.axhline(1.0, color="0.4", linestyle=":", label="Unweighted")
fid_ax.plot(time, abs(fid).data, color="0.4", linestyle=":", label="Unweighted")
for label, weighted, window in comparisons:
    (line,) = weights_ax.plot(time, window.data, label=label)
    fid_ax.plot(time, abs(weighted).data, color=line.get_color())

weights_ax.set(ylabel="Window weight")
weights_ax.legend()
fid_ax.set(xlabel="Acquisition time / ms", ylabel="FID magnitude / count")
plt.show()

# %% [raw] raw_mimetype="text/restructuredtext"
# Compare the spectra
# -------------------
#
# Use the same zero filling and phase correction for all four traces to
# isolate the effect of the window. Compare peak width, height and the
# surrounding baseline on the same intensity scale. The stronger exponential
# and Gaussian settings make the changes near the main resonance visible.

# %%
fig, ax = plt.subplots(layout="constrained")
datasets = [("Unweighted", fid)]
datasets.extend((label, weighted) for label, weighted, _ in comparisons)
for label, dataset in datasets:
    spectrum = dataset.zf_size(size=32768).fft().pk(phc0=52.44, phc1=-16.84)
    region = spectrum.real[2.0:-0.5]
    ax.plot(region.x.data, region.data, label=label)

ax.set(xlim=(2, -0.5), xlabel="Chemical shift / ppm", ylabel="Intensity / count")
ax.legend()
plt.show()

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
