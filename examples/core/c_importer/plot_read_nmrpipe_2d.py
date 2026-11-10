"""
Read a processed NMRPipe 2D COSY
================================

Open a real COSY plane of gossypol from
`nmrXiv sample S213 <https://nmrxiv.org/sample/S213>`_ (CC0 1.0).
The NMRPipe reference contains the processed real-real component converted
from TopSpin. The interop reader detects the profile from the header and
returns a real ``NDDataset`` with both frequency axes in hertz.
"""

# %%
import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/nmrpipe/nmrpipe_2d_cosy_example.ft2"
spectrum = scp.nmr.read(path, protocol="interop")
spectrum

# %%
# Display the spectrum as a contour map.
_ = spectrum.plot_map()
