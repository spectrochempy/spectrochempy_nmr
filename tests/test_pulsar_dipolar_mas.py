"""MAS dipolar physics against Cartesian trajectories and time integration."""

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.linalg import expm

from spectrochempy_nmr import pulsar

MAGIC_ANGLE = np.rad2deg(np.arccos(1 / np.sqrt(3)))
CARBON = pulsar.SpinSite(0.5, 100.6e6)
PROTON = pulsar.SpinSite(0.5, 400e6)
SETTINGS = {
    "observed": CARBON,
    "partner": PROTON,
    "dipolar_hz": 5000,
    "spectral_width_hz": 40000,
    "spinning_hz": 10000,
    "rotor_angle_deg": MAGIC_ANGLE,
    "accuracy": 2,
}
PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]) / 2
COPS = [np.kron(op, np.eye(2)) for op in PAULI]
HOPS = [np.kron(np.eye(2), op) for op in PAULI]
INITIAL = COPS[2] + PROTON.larmor_hz / CARBON.larmor_hz * HOPS[2]


@pytest.mark.parametrize("bond", [(0, 0, 1), np.array([1, 2, 3]) / np.sqrt(14)])
def test_magic_angle_refocuses_dipole_after_whole_rotor_periods(bond):
    events = [pulsar.IdealPulse(90)]
    reference = pulsar.simulate_coupled_sequence(
        events, **SETTINGS, dipolar_axis=bond, output="density"
    )
    density = pulsar.simulate_coupled_sequence(
        [*events, pulsar.Delay(3 / SETTINGS["spinning_hz"])],
        **SETTINGS,
        dipolar_axis=bond,
        output="density",
    )
    np.testing.assert_allclose(density, reference, atol=1e-13)
    fractional = pulsar.simulate_coupled_sequence(
        [*events, pulsar.Delay(0.4 / SETTINGS["spinning_hz"], decouple_j=True)],
        **SETTINGS,
        dipolar_axis=bond,
        output="density",
    )
    assert np.linalg.norm(fractional - reference) > 0.01


def test_rotor_parallel_to_field_recovers_static_cp():
    events = [
        pulsar.IdealPulse(0, partner_flip_deg=90),
        pulsar.Pulse(30e-6, 30000, 90, partner_rf_hz=30000, partner_phase_deg=90),
    ]
    static = pulsar.simulate_coupled_sequence(
        events, **{**SETTINGS, "spinning_hz": 0}, output="density"
    )
    spinning = pulsar.simulate_coupled_sequence(
        events, **{**SETTINGS, "rotor_angle_deg": 0}, output="density"
    )
    np.testing.assert_allclose(spinning, static, atol=1e-12)


def test_dipolar_sidebands_match_independent_time_domain_powder_average():
    rotor_hz, d = 10000, 16000
    spectrum = pulsar.simulate_coupled_sequence(
        [pulsar.IdealPulse(90)],
        **{**SETTINGS, "dipolar_hz": d, "spectral_width_hz": 140000, "accuracy": 8},
        points=14001,
        sidebands=6,
    )
    # Independent integration of a moving field on a cone. Axial symmetry
    # reduces the powder integral to a uniform integral over cos(beta).
    u, weights = np.polynomial.legendre.leggauss(120)
    time = np.linspace(0, 1 / rotor_hz, 4097)
    cos_theta = u[:, None] / np.sqrt(3) - np.sqrt(1 - u[:, None] ** 2) * np.sqrt(
        2 / 3
    ) * np.cos(2 * np.pi * rotor_hz * time)
    frequency = -d / 2 * (3 * cos_theta**2 - 1)
    phase = 2 * np.pi * cumulative_trapezoid(frequency, time, axis=1, initial=0)
    amplitudes = np.fft.fft(np.exp(1j * phase[:, :-1]), axis=1) / 4096
    expected = weights @ np.abs(amplitudes) ** 2 / 2
    # Average the two partner projections, whose harmonics have opposite signs.
    for order in range(-6, 7):
        bins = abs(spectrum.x.data - order * rotor_hz) < 20
        intensity = spectrum.data.real[bins].sum()
        reference = (expected[order] + expected[-order]) / 2
        assert intensity == pytest.approx(reference, abs=0.0015)
    represented = sum((expected[n] + expected[-n]) / 2 for n in range(-6, 7))
    assert spectrum.data.real.sum() == pytest.approx(represented, abs=2e-8)
    assert spectrum.data.imag.max() < 1e-12


@pytest.mark.parametrize("rotor_angle", [MAGIC_ANGLE, 39])
def test_finite_cp_and_delays_match_independent_cartesian_ode(monkeypatch, rotor_angle):
    # Fix one crystallite and rotor-phase quadrature to test the dynamics
    # without sharing the implementation's harmonic expansion or integrator.
    radial = np.array([2, -1, 3]) / np.sqrt(14)
    # Use the simulator's crystallite phase origin, expressed geometrically.
    polar = np.array(
        [
            radial[2] * radial[0],
            radial[2] * radial[1],
            -(radial[0] ** 2 + radial[1] ** 2),
        ]
    )
    polar /= np.linalg.norm(polar)
    azimuth = np.cross(radial, polar)
    bond = np.array([1, 2, -1]) / np.sqrt(6)
    monkeypatch.setattr(
        pulsar,
        "_powder_mesh",
        lambda *a, **k: (
            np.repeat(radial[:, None], 3, axis=1),
            np.ones(3),
            np.array([[0, 1, 2]]),
        ),
    )
    monkeypatch.setattr(pulsar, "_rf_rotor_count", lambda *a: 8)
    first = pulsar.Pulse(27e-6, 22000, 90, partner_rf_hz=30000, partner_phase_deg=90)
    second = pulsar.Pulse(31e-6, 28000, 0, partner_rf_hz=21000, partner_phase_deg=0)
    events = [
        pulsar.IdealPulse(0, partner_flip_deg=90),
        first,
        pulsar.Delay(37e-6),
        second,
    ]
    settings = {
        **SETTINGS,
        "dipolar_axis": bond,
        "j_hz": 173,
        "rotor_angle_deg": rotor_angle,
    }
    cone = np.deg2rad(rotor_angle)
    rotation = expm(-1j * np.pi / 2 * HOPS[0])
    initial = rotation @ INITIAL @ rotation.conj().T
    expected = np.zeros((4, 4), dtype=complex)
    for gamma in np.arange(8) * 2 * np.pi / 8:
        state, start = initial, 0
        for duration, rf in [
            (first.duration_s, 22000 * COPS[1] + 30000 * HOPS[1]),
            (37e-6, np.zeros((4, 4))),
            (second.duration_s, 28000 * COPS[0] + 21000 * HOPS[0]),
        ]:

            def derivative(t, flat, gamma=gamma, rf=rf):
                phi = gamma + 2 * np.pi * SETTINGS["spinning_hz"] * t
                field = radial * np.cos(cone) + np.sin(cone) * (
                    polar * np.cos(phi) + azimuth * np.sin(phi)
                )
                coupling = 173 + 5000 * (3 * (field @ bond) ** 2 - 1)
                hamiltonian = rf + coupling * (COPS[2] @ HOPS[2])
                rho = flat.reshape(4, 4)
                return (-2j * np.pi * (hamiltonian @ rho - rho @ hamiltonian)).ravel()

            solution = solve_ivp(
                derivative,
                (start, start + duration),
                state.ravel(),
                method="DOP853",
                rtol=1e-10,
                atol=1e-11,
            )
            assert solution.success
            state = solution.y[:, -1].reshape(4, 4)
            start += duration
        expected += state / 8
    coarse = pulsar.simulate_coupled_sequence(
        events, **settings, output="density", max_rf_step_s=1e-6
    )
    fine = pulsar.simulate_coupled_sequence(
        events, **settings, output="density", max_rf_step_s=0.25e-6
    )
    coarse_error = np.linalg.norm(coarse - expected)
    fine_error = np.linalg.norm(fine - expected)
    assert fine_error < coarse_error / 8
    np.testing.assert_allclose(fine, expected, atol=1e-4)


def test_two_channel_complex_contact_cache_preserves_density(monkeypatch):
    contact = pulsar.Pulse(20e-6, 22000, 37, partner_rf_hz=30000, partner_phase_deg=61)
    scans = tuple(
        (pulsar.IdealPulse(0, partner_flip_deg=90, partner_phase_deg=phase), contact)
        for phase in (0, 180)
    )
    cycle = pulsar.PhaseCycle(scans, (0.5, -0.5))
    count = 0
    propagate = pulsar._pulse_propagator

    def record(*args):
        nonlocal count
        count += 1
        return propagate(*args)

    monkeypatch.setattr(pulsar, "_pulse_propagator", record)
    settings = {**SETTINGS, "accuracy": 1, "output": "density"}
    cached = pulsar.simulate_coupled_sequence(cycle, **settings)
    cached_count = count
    count = 0
    monkeypatch.setattr(pulsar, "_RF_PROPAGATOR_CACHE_BYTES", 0)
    uncached = pulsar.simulate_coupled_sequence(cycle, **settings)
    assert count == 2 * cached_count
    np.testing.assert_allclose(cached, uncached, atol=1e-13)


@pytest.mark.parametrize(
    "carbon_rf,minimum", [(10000, 0.25), (20000, 0.45), (40000, 0.45), (50000, 0.25)]
)
def test_cp_mas_matches_first_and_second_rotor_sidebands(carbon_rf, minimum):
    scans = tuple(
        (
            pulsar.IdealPulse(0, partner_flip_deg=90, partner_phase_deg=phase),
            pulsar.Pulse(
                200e-6, carbon_rf, 90, partner_rf_hz=30000, partner_phase_deg=90
            ),
        )
        for phase in (0, 180)
    )
    density = pulsar.simulate_coupled_sequence(
        pulsar.PhaseCycle(scans, (0.5, -0.5)),
        **SETTINGS,
        output="density",
        max_rf_step_s=1e-6,
    )
    fraction = -np.trace(density @ COPS[1]).real / (PROTON.larmor_hz / CARBON.larmor_hz)
    assert fraction > minimum


def test_cp_mas_equal_rf_is_suppressed():
    scans = tuple(
        (
            pulsar.IdealPulse(0, partner_flip_deg=90, partner_phase_deg=phase),
            pulsar.Pulse(200e-6, 30000, 90, partner_rf_hz=30000, partner_phase_deg=90),
        )
        for phase in (0, 180)
    )
    cycle = pulsar.PhaseCycle(scans, (0.5, -0.5))
    density = pulsar.simulate_coupled_sequence(
        cycle,
        **SETTINGS,
        output="density",
        max_rf_step_s=1e-6,
    )
    fraction = -np.trace(density @ COPS[1]).real / (PROTON.larmor_hz / CARBON.larmor_hz)
    assert abs(fraction) < 1e-4


def test_cp_mas_supports_other_heteronuclear_larmor_ratios():
    scans = tuple(
        (
            pulsar.IdealPulse(0, partner_flip_deg=90, partner_phase_deg=phase),
            pulsar.Pulse(200e-6, 20000, 90, partner_rf_hz=30000, partner_phase_deg=90),
        )
        for phase in (0, 180)
    )
    cycle = pulsar.PhaseCycle(scans, (0.5, -0.5))
    fractions = []
    for observed, partner in [(CARBON, PROTON), (pulsar.SpinSite(0.5, 80e6), CARBON)]:
        density = pulsar.simulate_coupled_sequence(
            cycle,
            **{**SETTINGS, "observed": observed, "partner": partner},
            output="density",
            max_rf_step_s=1e-6,
        )
        fractions.append(
            -np.trace(density @ COPS[1]).real / (partner.larmor_hz / observed.larmor_hz)
        )
    assert fractions[0] > 0.45
    assert fractions[1] == pytest.approx(fractions[0], abs=1e-13)


def test_sum_frequency_mas_match_inverts_transferred_magnetization():
    # Low-power double-quantum match: nu1,C + nu1,H = nu_rotor.
    scans = tuple(
        (
            pulsar.IdealPulse(0, partner_flip_deg=90, partner_phase_deg=phase),
            pulsar.Pulse(200e-6, 4000, 90, partner_rf_hz=6000, partner_phase_deg=90),
        )
        for phase in (0, 180)
    )
    density = pulsar.simulate_coupled_sequence(
        pulsar.PhaseCycle(scans, (0.5, -0.5)),
        **SETTINGS,
        output="density",
        max_rf_step_s=1e-6,
    )
    fraction = -np.trace(density @ COPS[1]).real / (PROTON.larmor_hz / CARBON.larmor_hz)
    assert fraction < -0.05
