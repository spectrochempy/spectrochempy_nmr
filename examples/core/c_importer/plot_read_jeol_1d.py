"""
Read a JEOL 1D FID
==================

Open a proton FID from the epicatechin dataset in
`nmrXiv project P33 <https://doi.org/10.57992/nmrxiv.p33>`_ (CC0 1.0).
The reader returns the complex samples and their acquisition time axis
as an ``NDDataset``.
"""

# %%
import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/jeol/epicatechin_1h.jdf"
fid = scp.nmr.read_jeol(path)
fid

# %%
# Plot the real part of the imported FID. The dataset retains both real
# and imaginary samples for Fourier processing.
_ = fid.real.plot()
