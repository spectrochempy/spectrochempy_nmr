"""
Static ¹H–¹³C cross polarization
========================================

Transfer proton magnetization to carbon with simultaneous finite RF
contact pulses. The first plot varies contact time; the second shows
Hartmann–Hahn matching when the carbon RF crosses the proton RF.

This adapts historical PyPulsar's ``CP-contact.tpl`` and
``CP-HartmanHahn.tpl`` (5 kHz dipolar coupling, static powder).
It models one isolated spin pair with coherent oscillations, without a
proton bath, spin diffusion or rotating-frame relaxation. These curves
are not a bulk-sample CP build-up/decay fit.
"""

import numpy as np
import spectrochempy as scp
from spectrochempy.nmr import pulsar

# %%
# Observe carbon at a proton frequency of 400 MHz. D follows historical
# PyPulsar: H_dip/h = D * (3*cos(theta)**2 - 1) * Cz*Hz, in Hz.
# The default bond axis is the common sample-frame z axis. The powder
# integration averages its orientation relative to the static field.
carbon = pulsar.SpinSite(0.5, 100.6e6)
proton = pulsar.SpinSite(0.5, 400e6)
settings = {
    "observed": carbon,
    "partner": proton,
    "dipolar_hz": 5000,
    "spectral_width_hz": 20000,
    "accuracy": 8,
    "output": "density",
}
proton_rf = 50000


# %%
# Reverse only the initial proton 90° pulse and subtract the two scans.
# This isolates transferred proton magnetization and cancels any carbon
# signal created directly by the contact pulse. The preparatory rotation
# is ideal; the contact on both channels is a finite pulse.
def cp_program(contact_s, carbon_rf):
    scans = []
    for phase in (0, 180):
        events = [pulsar.IdealPulse(0, partner_flip_deg=90, partner_phase_deg=phase)]
        if contact_s:
            events.append(
                pulsar.Pulse(
                    contact_s,
                    carbon_rf,
                    90,
                    partner_rf_hz=proton_rf,
                    partner_phase_deg=90,
                )
            )
        scans.append(tuple(events))
    return pulsar.PhaseCycle(tuple(scans), (0.5, -0.5))


# Density matrices are raw arrays in the descending (carbon, proton)
# product basis. Project onto -Cy, the prepared proton spin-lock direction.
# Tr(Cy**2)=1 here. Divide by the initial proton/carbon thermal ratio,
# rather than normalizing each curve to its own maximum.
carbon_y = np.kron(np.array([[0, -1j], [1j, 0]]) / 2, np.eye(2))
thermal_ratio = proton.larmor_hz / carbon.larmor_hz


def transfer(contact_s, carbon_rf):
    density = pulsar.simulate_coupled_sequence(
        cp_program(contact_s, carbon_rf), **settings
    )
    return -np.trace(density @ carbon_y).real / thermal_ratio


# %%
# The matched and mismatched contacts share the same donor reference.
contacts = scp.Coord.linspace(0, 600, 31, units="us", title="Contact time")
contact_curves = [
    scp.NDDataset(
        [transfer(t * 1e-6, carbon_rf) for t in contacts.data],
        coordset=[contacts],
        title="Fraction of initial proton magnetization",
    )
    for carbon_rf in (50000, 35000)
]
_ = scp.plot_multiple(
    contact_curves,
    labels=["Matched: 50 / 50 kHz", "Mismatched: 35 / 50 kHz"],
    method="pen",
    legend="best",
    title="Static ¹H–¹³C CP — isolated pair, D = 5 kHz",
    xlabel="Contact time (µs)",
    ylabel="Transferred ¹H fraction",
)

# %%
# At a fixed 180 µs contact, sweep carbon RF across the 50 kHz proton RF.
carbon_fields = scp.Coord.linspace(35, 65, 41, units="kHz", title="Carbon RF")
matching = scp.NDDataset(
    [transfer(180e-6, rf * 1000) for rf in carbon_fields.data],
    coordset=[carbon_fields],
    title="Fraction of initial proton magnetization",
)
_ = matching.plot(
    title="Hartmann–Hahn matching\nProton RF = 50 kHz; contact = 180 µs",
    xlabel="Carbon RF (kHz)",
    ylabel="Transferred ¹H fraction",
)
