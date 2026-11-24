"""
Process a Multiplex SPAM spectrum
=================================

Run the public example through reading, common F2 phase correction and
shearing. See :doc:`/userguide/multiplex_spam` for the expected result and settings.

Data: Christian Fernandez, LCS, spectrochempy_data, CC BY 4.0.
"""

import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/RbNO3/206/ser"
if not path.exists():
    _ = scp.nmr.read(path, download_only=True)
raw = scp.nmr.read(path, phase_cycle="paired")
states = scp.nmr.Experiment(raw).select_coherence(3)
xf2 = scp.nmr.Experiment(states).xf2(size=4096)
phased = scp.nmr.Experiment(xf2).apk2d(order=0)
spectrum = scp.nmr.Experiment(phased).xfshear(size=512, f1_shift=7.0)
plane = scp.nmr.Experiment(spectrum).RR
_ = plane[:, -10.0:-60.0].plot(method="contour", start=0.1, nlevels=8)
