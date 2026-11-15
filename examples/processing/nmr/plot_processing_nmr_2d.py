"""
Compare indirect quadrature modes
=================================

Simulate one indirect resonance at a fixed F2 frequency. The five examples
show how F1 samples are reconstructed in QF, States, States-TPPI,
echo/anti-echo and TPPI. See :doc:`/userguide/processing_2d` for the
SpectroChemPy workflows on acquired data.

The example uses an explicit Fourier reference on arrays to isolate the
F1 recombination and bin selection. The magnitudes are normalized for
comparison, and the frequency axes are illustrative offsets in Hz.
"""

import numpy as np

import spectrochempy as scp

# %%
# Start with damped cosine and sine branches at 125 Hz.
size = 128
width = 1000.0
time = scp.Coord.linspace(0, size / width, size, endpoint=False, units="s")
angle = 2 * np.pi * 125.0 * time.data
decay = np.exp(-time.data / 0.030)
cosine = decay * np.cos(angle)
sine = decay * np.sin(angle)

# %%
# QF takes one branch. States reconstructs the complex pair.
# States-TPPI first undoes the sign alternation applied at acquisition.
# Bruker Rance–Kay echo/anti-echo first decodes the pathways into the pair.
alternating = (-1) ** np.arange(size)
states_tppi_cosine = alternating * cosine
states_tppi_sine = alternating * sine
echo = (cosine - 1j * sine) / 2
antiecho = (-cosine - 1j * sine) / 2
echo_cosine = echo - antiecho
echo_sine = 1j * (echo + antiecho)
signals = [
    cosine,
    cosine + 1j * sine,
    alternating * (states_tppi_cosine + 1j * states_tppi_sine),
    echo_cosine + 1j * echo_sine,
]
frequency = scp.Coord(
    np.fft.fftshift(np.fft.fftfreq(size, d=1 / width)),
    units="Hz",
    title="F1 offset",
)
spectra = []
for signal in signals:
    magnitude = abs(np.fft.fftshift(np.fft.fft(signal)))
    spectra.append(scp.NDDataset(magnitude, coordset=[frequency]))

# %%
# TPPI encodes a linear phase advance in a single real sequence.
# The simulated phase ramp moves the line by one quarter of the sampling
# frequency. Retain the positive half-band after zero filling and express
# its frequencies relative to that shift.
tppi = decay * np.cos(angle + np.pi / 2 * np.arange(size))
half_band = abs(np.fft.rfft(tppi, n=2 * size)[:size])
offset = scp.Coord.linspace(
    -width / 4, width / 4, size, endpoint=False, units="Hz", title="F1 offset"
)
spectra.append(scp.NDDataset(half_band, coordset=[offset]))

# %%
# The QF cosine has mirror peaks. The phase-sensitive combinations retain
# the sign of the simulated resonance.
_ = scp.multiplot(
    [spectrum / spectrum.max() for spectrum in spectra],
    labels=["QF", "States", "States-TPPI", "Echo/anti-echo", "TPPI"],
    nrow=5,
    sharex=True,
    method="pen",
    xlim=(-240, 240),
    ylim=(0, 1.05),
    ylabel="Magnitude",
    figsize=(7, 9),
)
