"""
Process an STMAS spectrum
=========================

Run the public example through reading, common F2 phase correction and
shearing. See :doc:`/userguide/stmas` for the expected result and settings.

Data: Christian Fernandez, LCS, spectrochempy_data, CC BY 4.0.
"""

import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/RbNO3/15/ser"
if not path.exists():
    _ = scp.nmr.read(path, download_only=True)
raw = scp.nmr.read(path, use_tdeff=True, indirect_dwell="IN[0]")
xf2 = scp.nmr.Experiment(raw).xf2(size=1024)
phased = scp.nmr.Experiment(xf2).apk2d(order=0)
spectrum = scp.nmr.Experiment(phased).xfshear(
    size=512, experiment="STMAS", f1_shift=-0.1
)
plane = scp.nmr.Experiment(spectrum).RR
_ = plane[:, -10.0:-60.0].plot(method="contour", start=0.1, nlevels=8)
