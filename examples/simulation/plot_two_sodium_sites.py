"""
Two sodium sites under MAS
==========================

Reuse one ideal pulse program for two independent sodium sites. Multiply
their complex spectra by relative populations of 0.75 and 1.0, then add
them. The real components and their sum share the same intensity scale.
"""

import spectrochempy as scp
from spectrochempy.nmr import pulsar

program = pulsar.load_sequence("ideal")
events = program.build({"a1": "90 deg", "ph1": 0})
settings = {
    "spin": 1.5,
    "larmor_hz": 105.804e6,
    "spinning_hz": 5000,
    "rotor_angle_deg": 54.735610,
    "spectral_width_hz": 4000000,
    "points": 131072,
    "sidebands": 8,
    "accuracy": 8,
    "transitions": "all",
}
sites = (
    {"cq_hz": 1500000, "eta": 0.3, "shift_hz": 211.608},
    {"cq_hz": 3000000, "eta": 1, "shift_hz": -2116.08},
)
weights = (0.75, 1.0)

# %%
# A wide computational window resolves the satellite rotor modulation.
# Display the central 30 kHz region, without normalising individual sites.
components = [
    weight * pulsar.simulate_pulse_sequence(events, **settings, **site)
    for weight, site in zip(weights, sites, strict=True)
]
total = components[0] + components[1]
for spectrum in (*components, total):
    spectrum.x.ito("kHz")

_ = scp.plot_multiple(
    [spectrum.real for spectrum in (*components, total)],
    labels=["Site 1 × 0.75", "Site 2 × 1.0", "Sum"],
    method="pen",
    legend="best",
    title="Two sodium sites — ideal excitation, 5 kHz MAS",
    xlabel="Frequency offset (kHz)",
    ylabel="Intensity per bin",
    xlim=(-15, 15),
    figsize=(11, 6),
)
