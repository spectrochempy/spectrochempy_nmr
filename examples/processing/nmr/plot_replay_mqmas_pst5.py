"""
Process an MQMAS spectrum
=========================

Run the public example through reading, common F2 phase correction and
shearing. See :doc:`/userguide/mqmas` for the expected result and settings.

Data: Christian Fernandez, LCS, spectrochempy_data, CC BY 4.0.
"""

import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/MQMAS_27AL_PST5/51/ser"
raw = scp.nmr.read(path)
xf2 = scp.nmr.Experiment(raw).xf2(size=4096, direct_points=2048)
phased = scp.nmr.Experiment(xf2).apk2d(order=0)
spectrum = scp.nmr.Experiment(phased).xfshear(
    size=512, experiment="3QMAS", f1_shift=-20.0
)
plane = scp.nmr.Experiment(spectrum).RR
_ = plane[52.0:25.0, 65.0:20.0].plot(method="contour", start=0.09, nlevels=8)
