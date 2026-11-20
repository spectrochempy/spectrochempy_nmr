"""Physical checks for a static, secular heteronuclear dipolar pair."""

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from spectrochempy_nmr import pulsar

CARBON = pulsar.SpinSite(0.5, 100.6e6)
PROTON = pulsar.SpinSite(0.5, 400e6)
SETTINGS = {
    "observed": CARBON,
    "partner": PROTON,
    "spectral_width_hz": 20000,
    "accuracy": 8,
}
PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]) / 2
CARBON_OPS = [np.kron(op, np.eye(2)) for op in PAULI]
PROTON_OPS = [np.kron(np.eye(2), op) for op in PAULI]


def cp_events(contact, carbon_rf=50000, proton_rf=50000):
    """Phase subtraction selects the initial proton magnetization only."""
    scans = tuple(
        (
            pulsar.IdealPulse(0, partner_flip_deg=90, partner_phase_deg=phase),
            pulsar.Pulse(
                contact, carbon_rf, 90, partner_rf_hz=proton_rf, partner_phase_deg=90
            ),
        )
        for phase in (0, 180)
    )
    return pulsar.PhaseCycle(scans, (0.5, -0.5))


def donor_fraction(density):
    return -np.trace(density @ CARBON_OPS[1]).real / (
        PROTON.larmor_hz / CARBON.larmor_hz
    )


def analytic_cp(contact, carbon_rf, proton_rf, dipolar_hz=5000, j_hz=0):
    # Exact two-spin solution, including counter-rotating double-quantum
    # terms in the spin-lock basis. Integrate uniformly over cos(theta),
    # independently of PULSAR's triangular powder grid and propagators.
    u, weights = np.polynomial.legendre.leggauss(300)
    g = (j_hz + dipolar_hz * (3 * u**2 - 1)) / 4
    minus = np.hypot((carbon_rf - proton_rf) / 2, g)
    plus = np.hypot((carbon_rf + proton_rf) / 2, g)
    transfer = g**2 * (
        (2 * np.pi * contact * np.sinc(2 * contact * minus)) ** 2
        - (2 * np.pi * contact * np.sinc(2 * contact * plus)) ** 2
    )
    return weights @ transfer / 2


@pytest.mark.parametrize("j_hz", [0, 1200])
def test_static_pake_profile_has_analytic_mass_support_and_second_moment(j_hz):
    d = 5000
    spectrum = pulsar.simulate_coupled_sequence(
        [pulsar.IdealPulse(90)], dipolar_hz=d, j_hz=j_hz, points=8193, **SETTINGS
    )
    # A spin-1/2 partner gives offsets +/-[J+D(3*u**2-1)]/2.
    mass = spectrum.data.real.copy()
    mass /= mass.sum()
    f = spectrum.x.data
    assert np.max(np.abs(spectrum.data.imag)) < 1e-12
    assert spectrum.data.real.sum() == pytest.approx(1)
    assert np.min(mass) >= -1e-12
    assert np.sum(np.abs(mass[np.abs(f) > d + abs(j_hz) / 2 + 5])) < 1e-12
    assert abs(f @ mass) < 3
    assert (f**2) @ mass == pytest.approx(d**2 / 5 + j_hz**2 / 4, rel=0.006)
    if j_hz == 0:
        assert abs(abs(f[mass.argmax()]) - d / 2) < 80


@pytest.mark.parametrize(
    "contact,carbon_rf", [(35e-6, 50000), (180e-6, 50000), (180e-6, 40000)]
)
def test_cp_matches_exact_two_spin_powder_solution(contact, carbon_rf):
    density = pulsar.simulate_coupled_sequence(
        cp_events(contact, carbon_rf), dipolar_hz=5000, output="density", **SETTINGS
    )
    expected = analytic_cp(contact, carbon_rf, 50000)
    assert donor_fraction(density) == pytest.approx(expected, abs=0.0015)
    np.testing.assert_allclose(density, density.conj().T, atol=1e-13)
    assert abs(np.trace(density)) < 1e-13


def test_cp_requires_dipolar_interaction_and_hartmann_hahn_matching():
    fractions = []
    for d, carbon_rf in [(0, 50000), (5000, 50000), (5000, 35000)]:
        density = pulsar.simulate_coupled_sequence(
            cp_events(180e-6, carbon_rf), dipolar_hz=d, output="density", **SETTINGS
        )
        fractions.append(donor_fraction(density))
    assert abs(fractions[0]) < 1e-12
    assert fractions[1] > 0.4
    assert abs(fractions[2]) < fractions[1] / 10


def test_density_and_spectrum_report_the_same_cp_amplitude():
    events = cp_events(180e-6)
    density = pulsar.simulate_coupled_sequence(
        events, dipolar_hz=5000, output="density", **SETTINGS
    )
    spectrum = pulsar.simulate_coupled_sequence(
        events, dipolar_hz=5000, points=2049, **SETTINGS
    )
    assert spectrum.data.sum().real == pytest.approx(
        -np.trace(density @ CARBON_OPS[1]).real, abs=1e-12
    )


def test_dipolar_echo_refocuses_with_partner_pi_but_j_decoupling_leaves_dipole():
    settings = {**SETTINGS, "output": "density", "dipolar_hz": 5000, "j_hz": 128}
    reference = pulsar.simulate_coupled_sequence([pulsar.IdealPulse(90)], **settings)
    echo = pulsar.simulate_coupled_sequence(
        [
            pulsar.IdealPulse(90),
            pulsar.Delay(70e-6),
            pulsar.IdealPulse(0, partner_flip_deg=180),
            pulsar.Delay(70e-6),
            pulsar.IdealPulse(0, partner_flip_deg=180),
        ],
        **settings,
    )
    np.testing.assert_allclose(echo, reference, atol=1e-12)
    decoupled = pulsar.simulate_coupled_sequence(
        [pulsar.IdealPulse(90), pulsar.Delay(140e-6, decouple_j=True)], **settings
    )
    dipole_only = pulsar.simulate_coupled_sequence(
        [pulsar.IdealPulse(90), pulsar.Delay(140e-6)], **{**settings, "j_hz": 0}
    )
    np.testing.assert_allclose(decoupled, dipole_only, atol=1e-12)
    assert np.linalg.norm(decoupled - reference) > 0.1


def test_arbitrary_bond_axis_matches_independent_spherical_integral():
    # Include a differently oriented CSA to make the relative bond frame
    # observable, and exercise all Cartesian cross terms of the dipole.
    bond = np.array([1, 2, 3]) / np.sqrt(14)
    rotation = Rotation.from_euler("xy", [31, 48], degrees=True).as_matrix()
    site = pulsar.SpinSite(
        0.5, 100.6e6, csa_anisotropy_hz=2200, csa_eta=0.4, csa_rotation=rotation
    )
    t = 73e-6
    actual = pulsar.simulate_coupled_sequence(
        [pulsar.IdealPulse(90), pulsar.Delay(t)],
        **{**SETTINGS, "observed": site, "accuracy": 12},
        dipolar_hz=5000,
        dipolar_axis=bond,
        output="density",
    )
    u, w = np.polynomial.legendre.leggauss(80)
    phi = np.arange(160) * (2 * np.pi / 160)
    x = np.sqrt(1 - u[:, None] ** 2) * np.cos(phi)
    y = np.sqrt(1 - u[:, None] ** 2) * np.sin(phi)
    z = np.broadcast_to(u[:, None], x.shape)
    n = np.stack((x, y, z), axis=-1)
    tensor = rotation @ np.diag([-1540, -660, 2200]) @ rotation.T
    shift = np.einsum("...i,ij,...j->...", n, tensor, n)
    coupling = 5000 * (3 * (n @ bond) ** 2 - 1)
    # Descending observed transitions use -shift*Iz in this engine.
    energies = -shift[..., None] * np.diag(CARBON_OPS[2])
    energies += coupling[..., None] * np.diag(CARBON_OPS[2] @ PROTON_OPS[2])
    initial = CARBON_OPS[2] + PROTON.larmor_hz / CARBON.larmor_hz * PROTON_OPS[2]
    pulse = expm(-1j * np.pi / 2 * CARBON_OPS[0])
    initial = pulse @ initial @ pulse.conj().T
    phases = np.exp(-2j * np.pi * t * energies)
    states = phases[..., :, None] * initial * phases[..., None, :].conj()
    expected = np.einsum("u,upij->ij", w / (2 * phi.size), states)
    np.testing.assert_allclose(actual, expected, atol=0.002)


@pytest.mark.parametrize(
    "axis", [(0, 0, 0), (1, 1, 1), (0, 1), (0, 0, np.nan), (0, 0, 1j)]
)
def test_invalid_bond_axis_is_rejected(axis):
    with pytest.raises(ValueError, match="dipolar_axis"):
        pulsar.simulate_coupled_sequence(
            [pulsar.IdealPulse(90)], dipolar_axis=axis, **SETTINGS
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"partner": CARBON}, "heteronuclear"),
        ({"partner": pulsar.SpinSite(1.5, 400e6)}, "spin-1/2"),
        ({"dipolar_hz": np.nan}, "finite"),
    ],
)
def test_unsupported_dipolar_models_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        pulsar.simulate_coupled_sequence(
            [pulsar.IdealPulse(90)], **{**SETTINGS, "dipolar_hz": 5000, **kwargs}
        )


def test_zero_dipolar_coupling_preserves_existing_j_result():
    events = [pulsar.Pulse(12e-6, 18000, partner_rf_hz=23000), pulsar.Delay(25e-6)]
    baseline = pulsar.simulate_coupled_sequence(events, j_hz=128, **SETTINGS)
    actual = pulsar.simulate_coupled_sequence(
        events, j_hz=128, dipolar_hz=0, **SETTINGS
    )
    np.testing.assert_array_equal(actual.data, baseline.data)
