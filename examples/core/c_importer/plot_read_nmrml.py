"""
Read and transform an nmrML FID
================================

Select the raw 1D profile and declare its point-count and sample convention.
Create the inputs once following :doc:`/userguide/reader_workflows`.
The nmrML reader requires the optional ``xmltodict`` package.
"""

import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/synthetic/signal.nmrML"
raw = scp.nmr.read(
    path,
    protocol="interop",
    profile="nmrml-1d",
    read_options={"raw_point_count": "scalar"},
    normalize_options={"conjugation": "none"},
)
spectrum = scp.nmr.Experiment(raw).process()
_ = spectrum.real.plot(xlim=(200, 800), xlabel="Frequency offset (Hz)")
