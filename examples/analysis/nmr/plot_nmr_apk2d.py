"""
Apply one automatic F2 phase to a spectral series
=================================================

Introduce the same 35-degree phase error into three synthetic spectra,
then estimate one correction from the first row and apply it to every row.
Generate the input as described in :doc:`/userguide/reader_workflows`.
See :doc:`/userguide/phase_2d` for common and individual row phases.
"""

import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/synthetic/simpson.fid"
reference = scp.nmr.Experiment(scp.nmr.read(path)).process()
shifted = reference.pk(phc0=35.0)
series = scp.NDDataset(
    [scale * shifted.data for scale in (1.0, 0.6, 0.3)],
    coordset=[scp.Coord([1, 2, 3], title="Trace"), shifted.x],
)
series.meta.isfreq = [False, True]
phased = scp.nmr.Experiment(series).apk2d(reference_row=1, order=0)
_ = scp.plot_multiple(
    [series[0].squeeze().real, phased[0].squeeze().real],
    labels=["Before", "After common F2 phase"],
    legend="best",
    xlim=(-800, -200),
)
