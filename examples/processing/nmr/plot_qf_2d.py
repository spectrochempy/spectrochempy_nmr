"""
Process a 2D QF spectrum
========================

Process the epicatechin raw JDF from the example data directory.
See :doc:`/userguide/qf_2d` for the processing details.
"""

import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/jeol/epicatechin_cosy.jdf"
raw = scp.nmr.read(path, quadrature="qf")
weighted = raw.sp(ssb=0, pow=2, dim="x").sp(ssb=0, pow=2, dim="y")
spectrum = scp.nmr.Experiment(weighted).process(size=(2048, 2048))
symmetric = scp.nmr.Experiment(abs(spectrum)).symmetrize()
_ = symmetric.plot(method="contour", xlabel="F2 (Hz)", ylabel="F1 (Hz)")
