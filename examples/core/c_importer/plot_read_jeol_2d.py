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
# Display the magnitude of the complex signal across all F1 increments
# and F2 time samples. The color scale shows the signal amplitude.
_ = abs(fid).plot(method="image", colorbar=True)
