"""
Read and transform a SIMPSON FID
================================

Read a synthetic text FID and display the transformed line in Hz.
Create the inputs once following :doc:`/userguide/reader_workflows`.
No SIMPSON executable is required to read its output format.
"""

import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/synthetic/simpson.fid"
raw = scp.nmr.read(path)
spectrum = scp.nmr.Experiment(raw).process()
_ = spectrum.real.plot(xlim=(-800, -200), xlabel="Frequency offset (Hz)")
