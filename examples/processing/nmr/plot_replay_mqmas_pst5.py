"""
Processing an MQMAS spectrum
============================

This example processes an aluminium-27 Bruker States acquisition of PST-5.
Estimate a common F2 phase, apply XFSHEAR and display the RR map on
isotropic F1 and direct F2 axes. The :doc:`MQMAS tutorial </userguide/mqmas>`
compares apodization and baseline correction on the same data.

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
# Estimate the zero-order phase from the 65–20 ppm region of the first
# complex F2 row.
experiment = scp.nmr.Experiment(xf2)
first_row = experiment.rsr(1)
preview = scp.nmr.apk(first_row[65.0:20.0], order=0)
if not preview.estimate.success:
    raise RuntimeError(preview.estimate.message)
_ = preview.phased.real.plot()

# %%
# Apply those same angles to both States branches at every F1 increment.
phase = {"phc0": preview.estimate.p0, "phc1": preview.estimate.p1, "pivot": 0}
phased = experiment.xf2p(**phase)

# %%
# XFSHEAR reconstructs the F2 quadrature and performs the F1 FFT. The
# optional shift moves the signal within the F1 window in isotropic ppm.
spectrum = scp.nmr.Experiment(phased).xfshear(size=512, spin=2.5, f1_shift=-20.0)

# %%
# Extract the RR plane. Use the calibrated ppm axes
# to select the region around the main resonances.
rr = scp.nmr.Experiment(spectrum).rr()
region = rr[52.0:25.0, 65.0:20.0]
_ = region.plot(method="contour", start=0.08, nlevels=12)
