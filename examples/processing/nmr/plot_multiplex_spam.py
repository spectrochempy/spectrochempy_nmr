"""
Reconstruct and shear a Multiplex SPAM spectrum
===============================================

Read the archived ⁸⁷Rb MQMultiplex32 acquisition and select 3Q.
Set ``SCP_MSM_DATA`` to the Bruker experiment directory, for example
``MULTIPLEX/4``. See :doc:`/userguide/multiplex_spam` for the acquisition
layout and the phase-cycle reconstruction.
"""

import os
from cmath import phase as phase_angle
from math import degrees

import spectrochempy as scp


path = os.environ.get("SCP_MSM_DATA")
if path:
    raw = scp.nmr.read(path, phase_cycle="paired")
    states = scp.nmr.Experiment(raw).select_coherence(3)
    xf2 = scp.nmr.Experiment(states).xf2(size=4096)
    row = xf2[0, 0].squeeze()
    phc0 = -degrees(phase_angle(row.data[abs(row).argmax()]))
    phased = scp.nmr.Experiment(xf2).xf2p(phc0=phc0)
    spectrum = scp.nmr.Experiment(phased).xfshear(size=256)
    _ = scp.nmr.Experiment(spectrum).RR.plot(method="contour")
else:
    print("Set SCP_MSM_DATA to the MQMultiplex32 experiment directory.")
