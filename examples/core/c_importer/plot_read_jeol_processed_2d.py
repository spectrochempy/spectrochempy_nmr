"""
Read a processed COSY export
============================

Read the epicatechin JCAMP-DX spectrum from the example data directory.
The file already contains a frequency-domain spectrum.
"""

import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/jeol/epicatechin_cosy.jdx"
spectrum = scp.nmr.read(path, protocol="interop", profile="jcampdx-spectrum-2d")
_ = spectrum.plot(method="contour", xlabel="F2 (Hz)", ylabel="F1 (Hz)")
