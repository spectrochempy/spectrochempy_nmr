"""
Simulate a quadrupolar powder line shape with PULSAR
======================================================

Simulate the central transition of a spin-5/2 nucleus under magic-angle
spinning and compare the spectrum before and after Gaussian broadening.
See the `PULSAR tutorial
<https://www.spectrochempy.fr/projects/nmr/latest/userguide/pulsar.html>`__ for the simulation conventions.
"""

import spectrochempy as scp
from spectrochempy.nmr import pulsar

# %%
# Include two spinning sidebands on either side of the central band.
spectrum = pulsar.simulate_central_transition(
    spin=2.5,
    cq_hz=4e6,
    eta=0.6,
    larmor_hz=100e6,
    spectral_width_hz=100e3,
    shift_hz=4500,
    spinning_hz=10e3,
    sidebands=2,
)
broadened = pulsar.broaden(spectrum, gaussian_fwhm_hz=300)
spectrum.x.ito("kHz")
broadened.x.ito("kHz")

# At 100 MHz, a frequency offset of 1 kHz corresponds to 10 ppm.
# The two axes share the simulation's zero-frequency reference.
_ = scp.plot_multiple(
    [spectrum, broadened],
    method="pen",
    labels=["Simulation", "Gaussian broadening (300 Hz FWHM)"],
    legend="best",
    ylabel="Intensity per bin",
    secondary_x=("Chemical shift / ppm", lambda khz: 10 * khz, lambda ppm: ppm / 10),
)
