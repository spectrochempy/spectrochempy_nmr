"""
Compare a Bruker 1D FFT with a TopSpin spectrum
===============================================

Transform a Bruker FID and compare the resulting real spectrum with the
stored TopSpin spectrum from the same acquisition. The bundled reference
uses no apodization. Its processing size sets the FFT grid for this example.
"""

# %%
import spectrochempy as scp

path = "nmrdata/bruker/tests/nmr/topspin_1d"
fid = scp.nmr.read_topspin(path, expno=1, remove_digital_filter=True)
reference = scp.nmr.read_topspin(path, expno=1, procno=1)

# %%
# Apply the recorded phase after the FFT. Scale each real spectrum by its
# own maximum to compare peak positions and shapes.
spectrum = scp.nmr.Experiment(fid).process(size=reference.shape[-1], phase="metadata")
calculated = spectrum.real.normalize()
topspin = reference.real.normalize()

_ = scp.plot_multiple(
    [topspin, calculated],
    method="pen",
    labels=["TopSpin", "SpectroChemPy FFT"],
    ls=["-", "--"],
    xlim=(-2.0, 12.0),
)
