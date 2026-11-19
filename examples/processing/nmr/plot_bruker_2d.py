"""
Transform a Bruker HMQC in both dimensions
==========================================

Read the PST-5 aluminium/phosphorus HMQC, apply squared cosine windows and
transform its States-TPPI components with the common processor. Display
the magnitude over all four components; retain them for phase-sensitive work.
See :doc:`/userguide/processing_2d` for filtering and component conventions.
The acquired prefix is described in :doc:`/userguide/bruker_indirect_sampling`.
Data: Christian Fernandez, LCS, CC BY 4.0, via spectrochempy_data.
"""

import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/topspin_2d/1/ser"
raw = scp.nmr.read(path, remove_digital_filter=True)
weighted = raw.sp(ssb=2, pow=2, dim="x").sp(ssb=2, pow=2, dim="y")
spectrum = scp.nmr.Experiment(weighted).process()
magnitude = (spectrum**2).sum(dim="c") ** 0.5
_ = magnitude.plot(
    method="contour",
    start=0.15,
    nlevels=12,
    negative=False,
    xlabel="F2 offset (Hz)",
    ylabel="F1 offset (Hz)",
    figsize=(7, 5),
)
