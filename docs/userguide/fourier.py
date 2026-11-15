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
# Fourier transformation in NMR
# =============================
#
# The core SpectroChemPy API provides ``fft()`` for transforming a sampled
# time axis and ``mc()`` for taking the magnitude of the complex result.
# Use ``dim="x"`` for the direct F2 dimension and ``dim="y"`` for F1 in
# the usual 2D layout. For acquired data, identify the indirect quadrature
# first. The :doc:`2D processing guide <processing_2d>` shows when a
# reader or processing workflow must reconstruct the branches.
#
# Fourier-transform a real signal
# -------------------------------
#
# Sample a 125 Hz cosine at 1,000 samples per second. The cosine frequency
# lies exactly on a Fourier bin. With 128 samples of unit amplitude, the
# unscaled forward transform gives two peaks of height 64 at ±125 Hz.
# This gives a known reference for checking the result.
#
# On this time-domain dataset, ``fft()`` returns a full centered complex
# spectrum even though the input is real. Its negative-frequency bins
# are the complex conjugates of the corresponding positive-frequency
# bins. For a one-sided display, slice the nonnegative frequencies by
# coordinate. With this even transform size the slice retains DC and
# excludes Nyquist. Then call ``mc()`` to display the modulus in each bin.

# %%
from cmath import exp
from math import cos, tau

import spectrochempy as scp

points = 128
width = 1000.0
time = scp.Coord.linspace(
    0, points / width, points, endpoint=False, units="s", title="F1 acquisition time"
)
cosine = [cos(tau * 125.0 * t) for t in time.data]
fid = scp.NDDataset(cosine, coordset=[time], meta={"td": [points]})

transformed = fid.fft()
magnitude = transformed.mc()
positive_magnitude = transformed[0.0:].mc()
_ = scp.multiplot(
    [magnitude, positive_magnitude],
    labels=["Full spectrum", "Nonnegative frequencies"],
    ncol=2,
    sharey=True,
    method="pen",
    ylabel="Magnitude",
    figsize=(9, 3),
)

# %% [raw] raw_mimetype="text/restructuredtext"
# Transform a 2D signal and calculate its magnitude
# -------------------------------------------------
#
# Give the same indirect cosine a complex F2 oscillation at 500 Hz.
# In this separable example there is one trace per F1 increment and
# there are no quadrature branches to decode. Transform F2 first.
# The F2 result is complex, so retain both its real and imaginary parts
# when transforming the indirect dimension. Take the magnitude only
# after both transforms.
#
# The expected peaks lie at F2 = 500 Hz and F1 = ±125 Hz. Taking the
# magnitude discards phase information and retains both mirror peaks.
# Use this example to check transform order and bin locations. For
# a QF workflow compared against NMRPipe reference data, see
# :doc:`../nmrpipe_interop`.

# %%
direct_time = scp.Coord.linspace(
    0, points / 4000, points, endpoint=False, units="s", title="F2 acquisition time"
)
direct_fid = [exp(1j * tau * 500.0 * t) for t in direct_time.data]
fid_2d = scp.NDDataset(
    [[weight * sample for sample in direct_fid] for weight in cosine],
    coordset=[time, direct_time],
    meta={"td": [points, points]},
)
f2 = fid_2d.fft(dim="x")
plane = f2.fft(dim="y").mc()
peak = plane.max()
levels = [peak * fraction for fraction in (0.15, 0.30, 0.50, 0.75)]
_ = plane.plot(
    method="contour",
    levels=levels,
    xlim=(250, 750),
    ylim=(-250, 250),
    xlabel="F2 / Hz",
    ylabel="F1 / Hz",
    title="Magnitude after both transforms",
)

# %% [raw] raw_mimetype="text/restructuredtext"
# Exploratory case: JEOL COSY
# ---------------------------
#
# The epicatechin JEOL JDF can be imported and transformed along F2.
# On this file, the NMRPipe converter
# `delta2pipe <https://spin.niddk.nih.gov/NMRPipe/ref/prog/delta2pipe.html>`_
# reports a real indirect axis and acquisition mode ``Magnitude``.
# This suggests testing a real F1 sampling model, retaining the complex
# F2 result through the F1 transform and applying magnitude at the end.
# A comparison with a spectrum processed in JEOL Delta is still needed
# to verify the F1 convention and the frequency referencing. Native F1
# processing remains unavailable for this profile.
