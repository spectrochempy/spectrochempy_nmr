"""
Transform F2 of a JEOL 2D COSY
==============================

Process the direct dimension of the epicatechin COSY acquisition from
`nmrXiv project P33 <https://doi.org/10.57992/nmrxiv.p33>`_ (CC0 1.0).
The result has a frequency axis in F2 and retains the indirect acquisition
time axis in F1. For this JEOL profile, processing stops after F2 because
the indirect quadrature convention has yet to be established.
"""

# %%
import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/jeol/epicatechin_cosy.jdf"
fid = scp.nmr.read_jeol(path)

# %%
# Apply an exponential window and Fourier-transform the direct axis.
windowed = scp.em(fid, lb=2.0, dim="x")
mixed = scp.fft(windowed, dim="x")
mixed

# %%
# Overlay the real F2 spectra for the first five F1 increments.
traces: scp.NDDataset = mixed[:5, :].real
_ = traces.plot(method="lines")
