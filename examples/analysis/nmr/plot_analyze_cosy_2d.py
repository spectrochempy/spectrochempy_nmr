"""
Find and integrate cross-peaks in a 2D COSY
===========================================

This workflow adapts the peak-picking and 2D integration examples from
`nmrglue-ng <https://nmrglue-ng.readthedocs.io/latest/how-to/quantification.html>`_.
It uses a real NMRPipe plane of gossypol from
`nmrXiv sample S213 <https://nmrxiv.org/sample/S213>`_ (CC0 1.0).
Both axes are calibrated in Hz.
"""

# %%
import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/nmrpipe/nmrpipe_2d_cosy_example.ft2"
spectrum = scp.nmr.read(path, protocol="interop")
_ = spectrum.plot(method="contour")

# %%
# Select F1 and F2 explicitly. The threshold is specific to the intensities
# of this dataset. The result reports peak locations on both frequency axes.
peaks = spectrum.find_peaks(dims=("y", "x"), height=1e6, as_result=True)
peaks

# %%
# Zoom in on one positive cross-peak. The region limits are given in Hz,
# so they remain meaningful if the point spacing changes.
region = spectrum[2250.0:2450.0, 880.0:1010.0]
_ = region.plot(method="contour")

# %%
# Integrate along F2 and then F1. The area includes both axis units.
area = region.trapezoid(dim="x").trapezoid(dim="y")
area
