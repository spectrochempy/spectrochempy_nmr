"""
Compare a 1D spectrum before and after SNIP
===========================================

Estimate a baseline on the real component of a phased spectrum.
See :doc:`/userguide/baseline` for choosing the width and checking broad peaks.
Data: Christian Fernandez, LCS, CC BY 4.0, via spectrochempy_data.
"""

import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/topspin_1d/1/fid"
raw = scp.nmr.read(path, remove_digital_filter=True)
spectrum = (
    scp.nmr.Experiment(raw)
    .process(
        apodization="em",
        lb=2.0,
        size=32768,
        phase="manual",
        phc0=52.44,
        phc1=-16.84,
    )
    .real
)
corrected = spectrum.snip(snip_width=50)
_ = scp.plot_multiple(
    [trace[9.0:-1.0] for trace in (spectrum, corrected, spectrum - corrected)],
    labels=["Before", "After SNIP", "Estimated baseline"],
    legend="upper left",
)
