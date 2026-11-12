"""
Processing an MQMAS spectrum
============================

This example processes an aluminium-27 Bruker States acquisition of PST-5.
It follows the TopSpin XFSHEAR workflow with manual F2 phasing and
compares the effects of apodization and baseline correction. It reconstructs
the direct quadrature with a Hilbert transform and rescales F1 to the
isotropic chemical-shift axis.

Data: Christian Fernandez, LCS, distributed through
`spectrochempy_data <https://github.com/spectrochempy/spectrochempy_data>`_
under `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_.
"""

# %%
import spectrochempy as scp

# sphinx_gallery_thumbnail_number = 2

# Keep the acquired points intact at import. xf2() will
# correct the digital group delay on the F2 frequency grid.
raw = scp.nmr.read("nmrdata/bruker/tests/nmr/MQMAS_27AL_PST5/51/ser")

# %%
# Retain the first 2048 complex points in F2 and zero-fill to 4096 points
# for a denser frequency grid before phase adjustment.
xf2 = scp.nmr.Experiment(raw).xf2(
    size=4096, direct_points=2048, correct_digital_filter=True
)

# %%
# Inspect the first acquired row. Adjust the phase parameters on this
# preview. XF2 has already compensated the digital group delay on the
# F2 frequency grid. The pivot is a sample index counted from the start of F2.
phase = {"phc0": -58.0, "phc1": 0.0, "pivot": 0}
first_row = xf2[0, 0, :].squeeze()
_ = first_row.pk(**phase).real[65.0:20.0].plot()

# %%
# Apply the selected phase to both States branches at every F1 increment.
phased = scp.nmr.Experiment(xf2).xf2p(**phase)

# %%
# XFSHEAR reconstructs the F2 quadrature and performs the F1 FFT. The
# optional shift moves the signal within the F1 window in isotropic ppm.
spectrum = scp.nmr.Experiment(phased).xfshear(size=512, spin=2.5, f1_shift=-20.0)

# %%
# The real part of the first channel is RR. Use the calibrated ppm axes
# to select the region around the main resonances.
rr = spectrum[0, :, :].squeeze().real
region = rr[52.0:25.0, 65.0:20.0]
_ = region.plot(method="contour", start=0.08, nlevels=12)


# %%
# Apply a Gaussian before the F2 FFT, centered on the expected echo
# in each row. In this shifted-echo experiment the echo moves by
# (19/12) × t1 along F2, in addition to the digital group delay.
# Both States branches at a given t1 receive the same window. The
# Gaussian standard deviation is 0.8 ms. Keep the phase and shear
# settings identical to isolate the effect of the window.
xf2_windowed = scp.nmr.Experiment(raw).xf2(
    size=4096, direct_points=2048, echo_window=(19 / 12, 0.0008)
)
phased_windowed = scp.nmr.Experiment(xf2_windowed).xf2p(**phase)
windowed = scp.nmr.Experiment(phased_windowed).xfshear(
    size=512, spin=2.5, f1_shift=-20.0
)
windowed_rr = windowed[0, :, :].squeeze().real[52.0:25.0, 65.0:20.0]

# %%
# Fit a linear baseline along F2 for each row of the real RR map, using
# regions on either side of the resonances. All three maps use the same
# contour levels so changes in intensity remain visible.
baseline = scp.Baseline(
    model="polynomial", order=1, ranges=[(21.0, 24.0), (61.0, 64.0)]
)
baseline.fit(windowed_rr)
corrected = windowed_rr - baseline.baseline

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
