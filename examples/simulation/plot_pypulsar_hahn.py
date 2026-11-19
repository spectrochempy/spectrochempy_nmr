"""
Hahn echo and T2 relaxation under MAS
=====================================

Simulate a Hahn echo on the central transition of a spin-5/2 aluminium
site under 10 kHz magic-angle spinning. Increasing the echo delay reduces
the signal through T2 relaxation. The four spectra share the same
intensity scale so that this attenuation remains visible.

Download :download:`hahn.seq <hahn.seq>` beside this script or notebook.
The sequence stays unchanged while its parameters vary in Python.

.. literalinclude:: hahn.seq
   :language: text
"""

from pathlib import Path

import spectrochempy as scp
from spectrochempy.nmr import pulsar

# %%
# Load the pulse program once. It contains the pulses, coherence selection
# and delays; their values are supplied below.
folder = Path(__file__).parent if "__file__" in globals() else Path.cwd()
program = pulsar.read_sequence(folder / "hahn.seq")

# %%
# Set the sample and independent pulse lengths, with explicit units.
us, ms, kHz = scp.ur.us, scp.ur.ms, scp.ur.kHz
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
    "spinning_hz": spinning.m_as("Hz"),
    "rotor_angle_deg": 54.735610,
    "spectral_width_hz": 10000,
    "transitions": "central",
    "points": 1024,
    "accuracy": 5,
}
parameters = {"p1": p90, "p2": p180, "rf1": rf, "ph1": 0, "ph2": 90, "l1": 1}

# %%
# Synchronise the echo with the rotor, accounting for finite pulse lengths.
# Reuse the same program for each delay, retaining the absolute intensity.
spectra = []
for rotor_count in rotor_counts:
    tau = rotor_count * rotor_period
    events = program.build(
        {**parameters, "d1": tau - p90 / 2 - p180 / 2, "d2": tau - p180 / 2}
    )
    spectrum = pulsar.simulate_pulse_sequence(events, t2_s=t2.m_as("s"), **settings)
    spectrum.x.ito("kHz")
    spectra.append(spectrum)

# %%
# Display the real component and zoom into the central-transition profile.
labels = [f"τ = {(count * rotor_period).m_as('ms'):.1f} ms" for count in rotor_counts]
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
