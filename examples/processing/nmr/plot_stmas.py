"""
Process a States STMAS spectrum
===============================

Read RbNO₃ experiment ``15`` from the usual SpectroChemPy ``datadir``.
This workflow uses its acquisition and processing status files.
See :doc:`/userguide/stmas` for pair selection, the shear and referencing.
"""

from cmath import phase as phase_angle
from math import degrees

import spectrochempy as scp


root = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/RbNO3"
states = scp.nmr.read(root / "15", use_tdeff=True, indirect_dwell="IN[0]")
xf2 = scp.nmr.Experiment(states).xf2(size=1024)
peak = xf2.data[abs(xf2).argmax()]
phased = scp.nmr.Experiment(xf2).xf2p(phc0=-degrees(phase_angle(peak)))
spectrum = scp.nmr.Experiment(phased).xfshear(
    size=512, experiment="STMAS", f1_shift=-0.1
)
view = scp.nmr.Experiment(spectrum)
_ = abs(view.RR + 1j * view.RI)[:, -10.0:-60.0].plot(method="contour")
