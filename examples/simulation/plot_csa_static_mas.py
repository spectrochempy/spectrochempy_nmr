"""
Chemical shift anisotropy in a static powder and under MAS
==========================================================

Use the same synthetic spin-1/2 site in both spectra. The static powder
pattern spans the principal shifts. Under magic-angle spinning, the
first-order CSA produces a centerband at the isotropic shift and a
series of sidebands separated by the spinning frequency.
"""

import spectrochempy as scp
from spectrochempy.nmr import pulsar

settings = dict(
    anisotropy_hz=-3500,
    eta=0.5,
    shift_hz=250,
    spectral_width_hz=20000,
    points=4096,
    accuracy=20,
)
static = pulsar.simulate_csa(**settings)
mas = pulsar.simulate_csa(**settings, spinning_hz=1500, sidebands=6)
spectra = [pulsar.broaden(spectrum, gaussian_fwhm_hz=40) for spectrum in (static, mas)]
for spectrum in spectra:
    spectrum.x.ito("kHz")

_ = scp.multiplot(
    spectra,
    labels=["Static powder", "MAS at 1.5 kHz"],
    nrow=2,
    sharex=True,
    method="pen",
    xlim=(-8, 8),
    xlabel="Frequency offset (kHz)",
    ylabel="Intensity per bin",
    figsize=(8, 5),
)
