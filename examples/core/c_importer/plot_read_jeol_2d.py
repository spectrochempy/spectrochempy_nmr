"""
Read a raw JEOL 2D COSY
=======================

Read the epicatechin COSY acquisition from
`nmrXiv project P33 <https://doi.org/10.57992/nmrxiv.p33>`_ (CC0 1.0).
Both axes are in the time domain. The direct axis F2 is ``x`` and the
indirect axis F1 is ``y``.
"""

# %%
import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/jeol/epicatechin_cosy.jdf"
fid = scp.nmr.read_jeol(path)
fid

# %%
# Inspect the first F1 increment as a direct-dimension FID. This slice
# illustrates the stored signal before any Fourier transform.
_ = fid[0, :].real.plot()
