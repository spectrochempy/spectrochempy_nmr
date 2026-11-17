"""
Export and reload a processed NMR spectrum
==========================================

Compare a synthetic real spectrum with its JCAMP-DX and CSDF round trips.
See the :doc:`exchange guide </userguide/exchanging_spectra>` for supported
components, coordinate units and metadata.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import spectrochempy as scp

frequency = scp.Coord.linspace(1000, -1000, 501, units="Hz", rounding=False)
spectrum = scp.lorentzian(frequency, pos=250, width=40, normalized=False)
spectrum.name = "Synthetic 1H spectrum"
with TemporaryDirectory() as directory:
    jdx_path = scp.nmr.write_jcamp(
        spectrum,
        Path(directory) / "spectrum.jdx",
        observation_frequency=400,
        nucleus="^1H",
    )
    csdf_path = scp.nmr.write_csdf(spectrum, Path(directory) / "spectrum.csdf")
    jcamp = scp.jcamp.read(jdx_path, jcamp_kind="nmr")
    csdf = scp.nmr.read_csdf(csdf_path)

_ = scp.plot_multiple(
    [spectrum, jcamp, csdf],
    labels=["Original", "JCAMP-DX", "CSDF"],
    method="pen",
    legend="best",
)
