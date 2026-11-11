"""
Processing an MQMAS spectrum
============================

Start from experiment 51 of the aluminium-27 Bruker States acquisition of PST-5. The
example uses the rectangular TopSpin recipe with no baseline correction
and manual F2 phase. It reconstructs the direct quadrature with a Hilbert
transform and rescales F1 to the isotropic chemical-shift axis.

Data: Christian Fernandez, LCS, distributed through
`spectrochempy_data <https://github.com/spectrochempy/spectrochempy_data>`_
under `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_.
"""

# %%
import spectrochempy as scp

raw = scp.nmr.read(
    "nmrdata/bruker/tests/nmr/MQMAS_27AL_PST5/51/ser",
    remove_digital_filter=False,
    remove_dc_offset=False,
)

# %%
# Retain the first 2048 complex points in F2 and zero-fill to 4096 points,
# as recommended by TopSpin for Hilbert reconstruction. Aluminium-27 has
# spin 5/2. For triple-quantum coherence, the signed slope is 19/12
# and the F1 scale divisor is 17/12.
xf2 = scp.nmr.Experiment(raw).prepare_xfshear(size=(512, 4096), direct_points=2048)

# %%
# Inspect the first acquired row. Adjust the phase parameters on this
# preview. The first-order starting value compensates the recorded
# 76-point digital group delay as a phase ramp on the F2 FFT grid. The
# pivot is a sample index counted from the start of F2.
phase = {"phc0": -58.0, "phc1": -27360.0, "pivot": 0}
first_row = xf2[0, 0, :].squeeze()
_ = first_row.pk(**phase).real[65.0:20.0].plot()

# %%
# Pass the same phase to XFSHEAR. It applies the correction to both
# States branches and every F1 increment before the shear and Hilbert
# reconstruction. XFSHEAR also performs the F1 FFT. The optional shift
# moves the signal within the F1 window in isotropic ppm.
# sphinx_gallery_thumbnail_number = 2
spectrum = scp.nmr.Experiment(xf2).replay_xfshear(
    spin=2.5, coherence_order=3, f1_shift=-20.0, **phase
)

# %%
# The real part of the first channel is RR. Use the calibrated ppm axes
# to select the region around the main resonances.
rr = spectrum[0, :, :].squeeze().real
region = rr[52.0:25.0, 65.0:20.0]
_ = region.plot(method="contour", start=0.08, nlevels=12)
