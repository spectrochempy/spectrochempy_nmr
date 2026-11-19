"""
Optimise a quadrupolar excitation pulse under MAS
=================================================

Sweep the pulse duration for a sodium site while keeping its RF power
fixed. The magnitude of the integrated central-transition signal shows
which durations efficiently excite this site. Every simulation reuses the
same rectangular pulse program from the sequence library.
"""

import spectrochempy as scp
from spectrochempy.nmr import pulsar

program = pulsar.load_sequence("rectangular")
settings = {
    "spin": 1.5,
    "larmor_hz": 105.804e6,
    "cq_hz": 2.5e6,
    "eta": 0,
    "shift_hz": 529.02,
    "spinning_hz": 10000,
    "spectral_width_hz": 50000,
    "transitions": "central",
    "sidebands": 1,
    "points": 2048,
    "accuracy": 4,
}
durations = scp.Coord.linspace(0.5, 9.5, 19, units="us", title="Pulse duration")
amplitudes = []
for duration in durations.data:
    events = program.build({"p1": duration * scp.ur.us, "rf1": "100 kHz", "ph1": 0})
    spectrum = pulsar.simulate_pulse_sequence(events, **settings)
    amplitudes.append(abs(spectrum.sum()))

response = scp.NDDataset(amplitudes, coordset=[durations])
_ = response.plot(
    title="Sodium excitation — 100 kHz RF, 10 kHz MAS",
    xlabel="Pulse duration (µs)",
    ylabel="Magnitude of integrated central-transition signal",
    figsize=(11, 6),
)
