"""
Correct a 2D baseline along F2 and F1
=====================================

Remove a known sloping baseline from a synthetic frequency-domain peak.
See :doc:`/userguide/baseline` for fit regions and interpretation.
Both commands return new datasets and operate on the RR plane.
"""

import spectrochempy as scp

f1 = scp.Coord.linspace(-500, 500, 101, units="Hz", title="F1")
f2 = scp.Coord.linspace(-1000, 1000, 201, units="Hz", title="F2")
indirect = scp.gaussian(f1, pos=0, width=100, normalized=False)
direct = scp.gaussian(f2, pos=0, width=120, normalized=False)
peak = indirect.data[:, None] * direct.data[None, :]
baseline = 0.2 + 0.0001 * f2.data[None, :] + 0.0002 * f1.data[:, None]
spectrum = scp.NDDataset(peak + baseline, coordset=[f1, f2])
spectrum.meta.isfreq = [True, True]
along_f2 = scp.nmr.Experiment(spectrum).abs2(
    order=1, ranges=[(-1000, -600), (600, 1000)]
)
corrected = scp.nmr.Experiment(along_f2).abs1(
    order=1, ranges=[(-500, -300), (300, 500)]
)
_ = scp.multiplot(
    [spectrum, corrected],
    method="image",
    labels=["Before", "After ABS2 + ABS1"],
    nrow=1,
    ncol=2,
    vmin=0,
    vmax=1.4,
    figsize=(9, 4),
)
