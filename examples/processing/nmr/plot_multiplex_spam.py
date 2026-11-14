"""
Reconstruct and shear a Multiplex SPAM spectrum
===============================================

Read the archived ⁸⁷Rb MQMultiplex32 acquisition and select 3Q.
Set ``SCP_MSM_DATA`` to the Bruker experiment directory, for example
``MULTIPLEX/4``. See :doc:`/userguide/multiplex_spam` for the acquisition
layout and the phase-cycle reconstruction.
"""

import os

import numpy as np
import spectrochempy as scp


path = os.environ.get("SCP_MSM_DATA")
if path:
    states = scp.nmr.read_multiplex_spam(path, coherence_order=3)
    xf2 = scp.nmr.Experiment(states).xf2(size=4096)
    row = xf2.data[0, 0]
    phc0 = -np.degrees(np.angle(row[np.argmax(np.abs(row))]))
    phased = scp.nmr.Experiment(xf2).xf2p(phc0=phc0)
    spectrum = scp.nmr.Experiment(phased).xfshear(size=256, spin=1.5, coherence_order=3)
    spectrum[0].squeeze().real.plot(method="contour")
else:
    print("Set SCP_MSM_DATA to the MQMultiplex32 experiment directory.")
