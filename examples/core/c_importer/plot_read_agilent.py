"""
Read and transform an Agilent/Varian FID
========================================

Process a synthetic 1D acquisition with a 400 MHz observation frequency.
Create the small input files once as described in
:doc:`/userguide/reader_workflows`; they are stored below the usual datadir.
"""

import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/synthetic/agilent.fid"
raw = scp.nmr.read(path)
spectrum = scp.nmr.Experiment(raw).process()
_ = spectrum.real.plot(xlim=(2.0, 0.5), title="Synthetic Agilent/Varian spectrum")
