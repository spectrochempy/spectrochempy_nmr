"""
Optimise 3QMAS excitation and conversion pulse lengths
======================================================

Vary each strong-pulse duration independently for a static sodium site.
The installed MQMAS program selects a triple-quantum pathway before
conversion to the detected central transition. Keep the RF powers fixed
and compare the magnitudes of the integrated signals on a common scale.
"""

import spectrochempy as scp
from spectrochempy.nmr import pulsar

program = pulsar.load_sequence("mqmas")
parameters = {
    "rf_exc": "100 kHz",
    "rf_conv": "100 kHz",
    "ph1": 0,
    "ph2": 0,
    "q": -3,
    "d0": "0 us",
}
settings = {
    "spin": 1.5,
    "larmor_hz": 105.804e6,
    "cq_hz": 1.5e6,
    "eta": 0,
    "spectral_width_hz": 20000,
    "transitions": "central",
    "points": 1024,
    "accuracy": 5,
}

# %%
# Scan excitation with p2 fixed, then conversion with p1 fixed. The same
# sequence is reused for both scans; all durations remain adjustable.
scans = (
    ("p1", scp.Coord.linspace(0.5, 9.5, 19, units="us"), {"p2": "2.7 us"}),
    ("p2", scp.Coord.linspace(0.25, 4.75, 19, units="us"), {"p1": "3 us"}),
)
responses = []
for parameter, durations, fixed in scans:
    amplitudes = []
    for duration in durations.data:
        events = program.build({**parameters, **fixed, parameter: duration * scp.ur.us})
        spectrum = pulsar.simulate_pulse_sequence(events, **settings)
        amplitudes.append(abs(spectrum.sum()))
    response = scp.NDDataset(amplitudes, coordset=[durations])
    responses.append(response)

_ = scp.plot_multiple(
    responses,
    labels=["Excitation p1 (p2 = 2.7 µs)", "Conversion p2 (p1 = 3 µs)"],
    method="pen",
    legend="best",
    title="3QMAS pulse optimisation — 100 kHz RF",
    xlabel="Pulse duration (µs)",
    ylabel="Magnitude of integrated central-transition signal",
    figsize=(11, 6),
)
