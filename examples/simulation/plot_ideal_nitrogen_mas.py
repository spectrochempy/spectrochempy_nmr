"""
An ideal-pulse nitrogen-14 MAS spectrum
=======================================

Excite all transitions of a spin-1 nitrogen site with an instantaneous
90° pulse. Magic-angle spinning produces sidebands separated by 10 kHz.
The pulse angle and phase are parameters of the reusable ideal program.
"""

from spectrochempy.nmr import pulsar

program = pulsar.load_sequence("ideal")
events = program.build({"a1": "90 deg", "ph1": 0})
settings = {
    "spin": 1,
    "larmor_hz": 57.792e6,
    "cq_hz": 200000,
    "eta": 0.3,
    "shift_hz": 57.792,
    "spinning_hz": 10000,
    "rotor_angle_deg": 54.735610,
    "spectral_width_hz": 1000000,
    "points": 8192,
    "sidebands": 20,
    "accuracy": 12,
    "transitions": "all",
}
spectrum = pulsar.simulate_pulse_sequence(events, **settings)
spectrum.x.ito("kHz")

_ = spectrum.real.plot(
    title="Nitrogen-14 — ideal excitation, 10 kHz MAS",
    xlabel="Frequency offset (kHz)",
    ylabel="Intensity per bin",
    xlim=(-150, 150),
    figsize=(11, 6),
)
