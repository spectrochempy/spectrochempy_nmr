"""
Finite-pulse excitation of a quadrupolar central transition
===========================================================

Compare four RF pulse lengths on an aluminium site under MAS. Load the
rectangular pulse program from the installed sequence library, then change
its parameters without editing the program. The real spectra retain their
common amplitude scale, without individual peak normalisation.
"""

import spectrochempy as scp
from spectrochempy.nmr import pulsar

program = pulsar.load_sequence("rectangular")
settings = {
    "spin": 2.5,
    "larmor_hz": 104.228e6,
    "cq_hz": 2.5e6,
    "eta": 0.3,
    "shift_hz": 208.456,
    "spinning_hz": 5000,
    "spectral_width_hz": 30000,
    "transitions": "central",
    "sidebands": 2,
    "points": 2048,
    "accuracy": 5,
}

# %%
# Bind durations and RF settings with explicit units. Only the pulse length
# changes between simulations.
durations_us = (0.5, 1, 2, 3)
spectra = []
for duration in durations_us:
    events = program.build({"p1": duration * scp.ur.us, "rf1": "100 kHz", "ph1": 0})
    spectrum = pulsar.simulate_pulse_sequence(events, **settings)
    spectrum.x.ito("kHz")
    spectra.append(spectrum.real)

_ = scp.plot_multiple(
    spectra,
    labels=[f"{duration:g} µs" for duration in durations_us],
    method="pen",
    legend="best",
    title="Finite RF excitation — 100 kHz RF, 5 kHz MAS",
    xlabel="Frequency offset (kHz)",
    ylabel="Intensity per bin",
    xlim=(-1, 0.4),
    figsize=(11, 6),
)
