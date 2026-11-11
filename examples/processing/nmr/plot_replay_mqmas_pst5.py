"""
Replay a 2D aluminium-27 MQMAS spectrum of PST-5
================================================

Start from experiment 51 of the PST-5 Bruker States acquisition. The
example uses the rectangular TopSpin recipe with no baseline correction
or F1 shift. It reconstructs the direct quadrature with a Hilbert
transform and rescales F1 to the isotropic chemical-shift axis.

To run the notebook, set the environment variable ``SCP_MQMAS_SER`` to
the path of your PST-5 experiment 51 SER file, with its acquisition and
processing parameters in the original Bruker directory structure.
The acquisition is supplied locally.
"""

# %%
import os
from pathlib import Path

import spectrochempy as scp

ser_path = os.environ.get("SCP_MQMAS_SER")
if not ser_path:
    print("Set SCP_MQMAS_SER to the path of the PST-5 experiment 51 SER file.")
else:
    raw = scp.nmr.read_topspin(
        Path(ser_path),
        local_only=True,
        remove_digital_filter=False,
        remove_dc_offset=False,
    )

# %%
# Retain the first 2048 complex points in F2 and zero-fill to 4096 points,
# as recommended by TopSpin for Hilbert reconstruction. Aluminium-27 has
# spin 5/2. For triple-quantum coherence, the signed slope is 19/12
# and the F1 scale divisor is 17/12.
if ser_path:
    spectrum = scp.nmr.Experiment(raw).replay_xfshear(
        spin=2.5, coherence_order=3, size=(512, 4096), direct_points=2048
    )

# %%
# The real part of the first channel is RR. Use the calibrated ppm axes
# to select the region around the main resonances.
if ser_path:
    rr = spectrum[0, :, :].squeeze().real
    region = rr[55.0:25.0, 65.0:20.0]
    _ = region.plot(method="contour", start=0.08, nlevels=12)
