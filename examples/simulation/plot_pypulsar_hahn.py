"""
Hahn echo and T2 relaxation under MAS
=====================================

Simulate a Hahn echo on the central transition of a spin-5/2 aluminium
site under 10 kHz magic-angle spinning. Increasing the echo delay reduces
the signal through T2 relaxation. The four spectra share the same
intensity scale so that this attenuation remains visible.
"""

import spectrochempy as scp
from spectrochempy.nmr import pulsar

# %%
# Define the site and the finite RF pulses. The scale factors keep durations
# readable in microseconds while passing seconds to the simulation engine.
us, ms, kHz = 1e-6, 1e-3, 1e3
spinning = 10 * kHz
rotor_period = 1 / spinning
rf = 10 * kHz
p90 = (25 / 3) * us
p180 = (50 / 3) * us
t2 = 1 * ms
rotor_counts = (1, 6, 11, 16)
settings = {
    "spin": 2.5,
    "larmor_hz": 104.228e6,
    "cq_hz": 2.5e6,
    "eta": 0.3,
    "spinning_hz": spinning,
    "rotor_angle_deg": 54.735610,
    "spectral_width_hz": 10 * kHz,
    "transitions": "central",
    "points": 1024,
    "accuracy": 5,
}

# %%
# Synchronise the echo with the rotor, accounting for the finite pulse
# durations. Order -1 before the refocusing pulse leads to the detected
# +1 coherence after the y180 pulse.


def hahn_events(rotor_count):
    """Build one rotor-synchronised Hahn echo."""
    tau = rotor_count * rotor_period
    return (
        pulsar.Pulse(p90, rf),
        pulsar.Coherence(-1),
        pulsar.Delay(tau - p90 / 2 - p180 / 2),
        pulsar.Pulse(p180, rf, phase_deg=90),
        pulsar.Delay(tau - p180 / 2),
    )


spectra = []
for rotor_count in rotor_counts:
    spectrum = pulsar.simulate_pulse_sequence(
        hahn_events(rotor_count), t2_s=t2, **settings
    )
    spectrum.x.ito("kHz")
    spectra.append(spectrum)

# %%
# Display the real component and zoom into the central-transition profile.
# Longer delays produce weaker echoes; no individual normalisation is used.
labels = [f"τ = {count * rotor_period / ms:.1f} ms" for count in rotor_counts]
_ = scp.plot_multiple(
    [spectrum.real for spectrum in spectra],
    labels=labels,
    method="pen",
    legend="best",
    title="Hahn echo under MAS — T2 = 1 ms",
    xlabel="Frequency offset (kHz)",
    ylabel="Intensity per bin",
    xlim=(-1.2, 0.25),
    figsize=(11, 6),
)
