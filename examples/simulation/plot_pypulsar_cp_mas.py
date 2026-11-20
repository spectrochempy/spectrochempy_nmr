"""
Cross polarization under MAS
========================================

Transfer magnetization in an isolated ¹H–¹³C pair under 10 kHz
magic-angle spinning. Unlike the static example, the first-order
Hartmann–Hahn matches occur at RF differences of one or two rotor
frequencies. Equal RF amplitudes no longer give the static CP transfer.

Both spins are general spin-½ sites: the isotope labels here identify
the example's Larmor frequencies. The dipolar model also accepts other
heteronuclear spin-½ pairs. It does not include a proton bath, spin
diffusion, homonuclear couplings or rotating-frame relaxation.
"""

import numpy as np
import spectrochempy as scp
from spectrochempy.nmr import pulsar

# %%
# D uses historical PyPulsar's signed convention:
# H_dip(t)/h = D * (3*cos(theta(t))**2 - 1) * Cz*Hz, in Hz.
# Use the exact magic angle here to remove the rank-two mean interaction.
carbon = pulsar.SpinSite(0.5, 100.6e6)
proton = pulsar.SpinSite(0.5, 400e6)
rotor_hz = 10000
proton_rf = 30000
settings = {
    "observed": carbon,
    "partner": proton,
    "dipolar_hz": 5000,
    "spinning_hz": rotor_hz,
    "rotor_angle_deg": np.rad2deg(np.arccos(1 / np.sqrt(3))),
    "spectral_width_hz": 40000,
    "accuracy": 2,
    "max_rf_step_s": 1e-6,
    "output": "density",
}


# %%
# Subtract scans with opposite initial proton magnetization to isolate
# the transfer. The proton preparation is ideal; both RF contacts are
# finite. Rotor modulation continues throughout each contact pulse.
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


# Project the raw product density matrix onto the spin-lock direction,
# -Cy. Tr(Cy**2)=1. All curves share the initial proton magnetization
# reference, obtained from the thermal Larmor-frequency ratio.
carbon_y = np.kron(np.array([[0, -1j], [1j, 0]]) / 2, np.eye(2))
thermal_ratio = proton.larmor_hz / carbon.larmor_hz


def transfer(contact_s, carbon_rf):
    density = pulsar.simulate_coupled_sequence(
        cp_program(contact_s, carbon_rf), **settings
    )
    return -np.trace(density @ carbon_y).real / thermal_ratio


# %%
# Compare a first rotor-sideband match with equal RF amplitudes. This
# compact demonstration uses a coarse powder grid. Increasing accuracy
# also increases the number of initial rotor phases. Converge both
# accuracy and max_rf_step_s for the particular contact and RF range.
contacts = scp.Coord.linspace(0, 200, 9, units="us", title="Contact time")
contact_curves = [
    scp.NDDataset(
        [transfer(t * 1e-6, carbon_rf) for t in contacts.data],
        coordset=[contacts],
        title="Fraction of initial proton magnetization",
    )
    for carbon_rf in (20000, 30000)
]
_ = scp.plot_multiple(
    contact_curves,
    labels=["First MAS match: 20 / 30 kHz", "Equal RF: 30 / 30 kHz"],
    method="pen",
    legend="best",
    title="¹H–¹³C CP at 10 kHz MAS — isolated pair, D = 5 kHz",
    xlabel="Contact time (µs)",
    ylabel="Transferred ¹H fraction",
)

# %%
# Sweep carbon RF at a 200 µs contact (two rotor periods). Matches lie
# near 10, 20, 40 and 50 kHz for the 30 kHz proton field. This example
# samples the zero-quantum matches; sum-frequency matches can also occur
# at lower RF powers. No MAS-dependent RF condition is imposed in code:
# the curves follow from the time-dependent dipolar Hamiltonian.
carbon_fields = scp.Coord.linspace(5, 55, 21, units="kHz", title="Carbon RF")
matching = scp.NDDataset(
    [transfer(200e-6, rf * 1000) for rf in carbon_fields.data],
    coordset=[carbon_fields],
    title="Fraction of initial proton magnetization",
)
_ = matching.plot(
    title="Hartmann–Hahn matching at 10 kHz MAS\nProton RF = 30 kHz; contact = 200 µs",
    xlabel="Carbon RF (kHz)",
    ylabel="Transferred ¹H fraction",
)
