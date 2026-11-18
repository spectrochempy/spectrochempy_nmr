"""
Process a States STMAS spectrum
===============================

Set ``SCP_RBNO3_DATA`` to the local RbNO₃ directory containing experiment
``15``. This workflow uses its acquisition and processing status files.
See :doc:`/userguide/stmas` for pair selection, the shear and referencing.
"""

import os
from cmath import phase as phase_angle
from math import degrees
from pathlib import Path

import spectrochempy as scp


root = os.environ.get("SCP_RBNO3_DATA")
if root:
    states = scp.nmr.read(Path(root) / "15", use_tdeff=True, indirect_dwell="IN[0]")
    xf2 = scp.nmr.Experiment(states).xf2(size=1024)
    peak = xf2.data[abs(xf2).argmax()]
    phased = scp.nmr.Experiment(xf2).xf2p(phc0=-degrees(phase_angle(peak)))
    spectrum = scp.nmr.Experiment(phased).stmas_shear(size=512, f1_shift=0.1)
    spectrum.abs()[:, -10.0:-60.0].plot(method="contour")
else:
    print("Set SCP_RBNO3_DATA to the directory containing RbNO₃/15.")
