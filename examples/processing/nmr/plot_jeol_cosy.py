"""
Process a complete JEOL COSY spectrum
=======================================

Read the epicatechin COSY from
`nmrXiv P33 <https://doi.org/10.57992/nmrxiv.p33>`_ (CC0 1.0), transform
both dimensions and plot the 2D magnitude spectrum.
See :doc:`/userguide/jeol_cosy_magnitude` for processing details.
"""

# %%
import spectrochempy as scp
from spectrochempy_nmr import process_jeol_cosy_magnitude

path = scp.preferences.datadir / "nmrdata/jeol/epicatechin_cosy.jdf"
cosy = process_jeol_cosy_magnitude(path, f1_size=2048)
_ = cosy.plot(method="contour", start=0.02, nlevels=10, negative=False, cmap="viridis")
