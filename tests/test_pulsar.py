"""Regression checkpoints for the NumPy port of Masaï PULSAR."""

import numpy as np
import pytest

from spectrochempy.core.units import ur
from spectrochempy_nmr.pulsar import Coherence
from spectrochempy_nmr.pulsar import Delay
from spectrochempy_nmr.pulsar import IdealPulse
from spectrochempy_nmr.pulsar import PassiveJ
from spectrochempy_nmr.pulsar import Pulse
from spectrochempy_nmr.pulsar import SpinSite
from spectrochempy_nmr.pulsar import simulate_central_transition
from spectrochempy_nmr.pulsar import simulate_coupled_sequence
from spectrochempy_nmr.pulsar import simulate_csa
from spectrochempy_nmr.pulsar import simulate_pulse_sequence
from spectrochempy_nmr.pulsar import simulate_quadrupolar
from spectrochempy_nmr.pulsar import simulate_rectangular_pulse


@pytest.mark.parametrize(
    ("coupling_scale", "peak", "intensity", "centroid", "second_moment"),
    [
        (5000, 3957, 0.0220796894, -3.4806, 561779.33),
        (15000, 4568, 0.0186302159, -10.6250, 5057699.15),
        (25000, 4884, 0.0104423752, -36.4255, 14061097.23),
    ],
)
def test_archived_pulsar_mas_profile(
    coupling_scale, peak, intensity, centroid, second_moment
):
    # masai/masai/core/fit/pulsar/res.npy, generated parameters from pypulsar.py
    # SHA-256: 4851c192867e24c194815768fcd649186cf4c4ffaa46cb0d29b10128c08854f3
    dataset = simulate_central_transition(
        cq_hz=np.sqrt(12 * coupling_scale * 100e6),
        eta=0,
        larmor_hz=100e6,
        spectral_width_hz=50000,
        shift_hz=0.3 * coupling_scale,
        spinning_hz=50000,
    )
    frequency, profile = dataset.x.data, dataset.data
    assert dataset.x.units == ur.Hz
    np.testing.assert_allclose(frequency, np.linspace(-25000, 25000, 8192), atol=5e-4)
    assert np.all(profile >= 0)
    assert profile.sum() == pytest.approx(1)
    assert profile.argmax() == peak
    assert profile[peak] == pytest.approx(intensity, abs=5e-7)
    assert profile @ frequency == pytest.approx(centroid, abs=0.01)
    assert profile @ (frequency**2) == pytest.approx(second_moment, rel=1e-5)


def test_zero_coupling_places_all_intensity_at_the_shift():
    dataset = simulate_central_transition(
        cq_hz=0,
        eta=0.5,
        larmor_hz=100e6,
        spectral_width_hz=10000,
        shift_hz=1000,
        points=11,
        accuracy=1,
    )
    assert dataset.x.data[dataset.data.argmax()] == 1000
    assert np.count_nonzero(dataset.data) == 1
    assert dataset.data.sum() == pytest.approx(1)


@pytest.mark.parametrize("integer_type", [np.int32, np.uint32])
def test_coupling_scalar_dtype_does_not_change_profile(integer_type):
    params = dict(eta=0.8, larmor_hz=100e6, spectral_width_hz=50e3, accuracy=4)
    expected = simulate_central_transition(cq_hz=1_000_000, **params)
    actual = simulate_central_transition(cq_hz=integer_type(1_000_000), **params)
    np.testing.assert_array_equal(actual.data, expected.data)


def test_axial_static_powder_matches_analytic_centroid():
    # For eta=0, the static spin-3/2 central-transition offset is
    # 3*Cq**2/(64*nu0) * (1-u**2)*(1-9*u**2), with u=cos(theta).
    # Averaging over uniform u gives -Cq**2/(40*nu0).
    cq_hz, larmor_hz = 4e6, 100e6
    dataset = simulate_central_transition(
        cq_hz=cq_hz, eta=0, larmor_hz=larmor_hz, spectral_width_hz=40e3
    )
    centroid = dataset.data @ dataset.x.data
    # Allow for the powder mesh and the left-edge frequency convention.
    assert centroid == pytest.approx(-(cq_hz**2) / (40 * larmor_hz), abs=10)


def test_triangle_batches_preserve_flat_sides_and_endpoint_mass():
    from spectrochempy_nmr.pulsar import _deposit_triangles

    axis = np.arange(8192, dtype=float)
    # Equal rising and falling densities sum to a uniform distribution.
    # The last triangle is a point mass at the upper endpoint.
    frequencies = np.array([axis[0], axis[-1]])
    triangles = np.array([[0, 0, 1], [0, 1, 1]] * 33 + [[1, 1, 1]])
    result = _deposit_triangles(axis, frequencies, np.ones(2), triangles)
    expected = np.full(axis.size, 198 / (axis.size - 1))
    expected[-1] = 3
    np.testing.assert_allclose(result, expected, rtol=1e-11)


def test_mas_rejects_band_intensity_table_above_memory_limit():
    with pytest.raises(
        ValueError, match="MAS band intensities exceed the 64 MiB limit"
    ):
        simulate_central_transition(
            cq_hz=1e5,
            eta=0,
            larmor_hz=100e6,
            spectral_width_hz=41e3,
            spinning_hz=10.1,
            accuracy=64,
            sidebands=2000,
        )


def test_mas_zero_coupling_has_no_sidebands():
    dataset = simulate_central_transition(
        cq_hz=0,
        eta=0,
        larmor_hz=100e6,
        spectral_width_hz=60e3,
        spinning_hz=10e3,
        sidebands=2,
    )
    assert np.count_nonzero(dataset.data > 1e-14) == 1
    assert dataset.data.max() == pytest.approx(1)


def test_mas_sidebands_match_asymmetric_fortran_reference():
    dataset = simulate_central_transition(
        cq_hz=np.sqrt(12 * 15000 * 100e6),
        eta=0.7,
        larmor_hz=100e6,
        spectral_width_hz=100e3,
        shift_hz=4500,
        spinning_hz=10e3,
        sidebands=2,
    )
    # Fixed windows across the historical Fortran profile, normalized together.
    windows = [-np.inf, -15000, -5000, 5000, 15000, np.inf]
    areas, _ = np.histogram(dataset.x.data, bins=windows, weights=dataset.data)
    np.testing.assert_allclose(
        areas,
        [0.02150456, 0.19630213, 0.63106285, 0.10237457, 0.04875589],
        atol=3e-6,
        rtol=0,
    )


def _perturbative_level_offsets(directions, cq_hz, eta, larmor_hz, spin):
    # Work directly in the quadrupolar principal axes. Diagonalize the
    # Zeeman term for each field direction and apply second-order
    # perturbation theory to the central pair of Zeeman levels.
    m = np.arange(-spin, spin + 1)
    levels = m.size
    spin_squared = spin * (spin + 1)
    raising = np.diag(np.sqrt(spin_squared - m[:-1] * (m[:-1] + 1)), k=-1)
    ix = (raising + raising.T) / 2
    iy = (raising - raising.T) / (2j)
    iz = np.diag(m)
    quadrupolar = (
        cq_hz
        / (4 * spin * (2 * spin - 1))
        * (3 * (iz @ iz) - spin_squared * np.eye(levels) + eta * (ix @ ix - iy @ iy))
    )

    operators = np.stack((ix, iy, iz))
    field_operator = (directions @ operators.reshape(3, -1)).reshape(
        *directions.shape[:-1], levels, levels
    )
    zeeman = -larmor_hz * field_operator
    energies, basis = np.linalg.eigh(zeeman)
    matrix_elements = basis.conj().swapaxes(-1, -2) @ quadrupolar @ basis
    gaps = energies[..., :, None] - energies[..., None, :]
    gaps[..., np.arange(levels), np.arange(levels)] = np.inf
    corrections = (np.abs(matrix_elements) ** 2 / gaps).sum(axis=-1)
    return matrix_elements.diagonal(axis1=-2, axis2=-1).real, corrections


def _second_order_offsets(directions, cq_hz, eta, larmor_hz, spin):
    _, second = _perturbative_level_offsets(directions, cq_hz, eta, larmor_hz, spin)
    center = second.shape[-1] // 2
    return second[..., center] - second[..., center - 1]


@pytest.mark.parametrize("spin", [1.5, 2.5, 3.5])
def test_nonaxial_static_powder_matches_spin_hamiltonian_moments(spin):
    cq_hz, larmor_hz, eta = 4e6, 100e6, 0.7
    dataset = simulate_central_transition(
        cq_hz=cq_hz, eta=eta, larmor_hz=larmor_hz, spectral_width_hz=60e3, spin=spin
    )

    # Independent powder average with Gauss-Legendre weights in cos(theta)
    # and uniform azimuth samples, over the whole sphere.
    cosine, weights = np.polynomial.legendre.leggauss(48)
    azimuth = 2 * np.pi * np.arange(96) / 96
    z, phi = np.meshgrid(cosine, azimuth, indexing="ij")
    sine = np.sqrt(1 - z**2)
    directions = np.stack((sine * np.cos(phi), sine * np.sin(phi), z), axis=-1)
    shifts = _second_order_offsets(directions, cq_hz, eta, larmor_hz, spin)
    reference_mean = weights @ shifts.mean(axis=1) / 2
    reference_std = np.sqrt(weights @ ((shifts - reference_mean) ** 2).mean(axis=1) / 2)

    # Use bin centers for moments of the binned profile.
    axis = dataset.x.data
    centers = axis + (axis[-1] - axis[0]) / (2 * (axis.size - 1))
    mean = dataset.data @ centers
    std = np.sqrt(dataset.data @ ((centers - mean) ** 2))
    assert mean == pytest.approx(reference_mean, abs=3)
    assert std == pytest.approx(reference_std, abs=6)


@pytest.mark.parametrize(
    ("spin", "with_csa"), [(1.5, False), (2.5, False), (1.5, True)]
)
def test_mas_sideband_areas_match_time_domain_spin_hamiltonian(spin, with_csa):
    cq_hz, larmor_hz, eta, spinning_hz = 2.8e6, 100e6, 0.7, 10e3
    # Two successive rotations give a CSA tensor with three off-diagonals.
    rotation = np.array([[0.8, -0.6, 0], [0.6, 0.8, 0], [0, 0, 1]]) @ np.array(
        [[0.6, 0, 0.8], [0, 1, 0], [-0.8, 0, 0.6]]
    )
    csa = (
        dict(csa_anisotropy_hz=1200, csa_eta=0.4, csa_rotation=rotation)
        if with_csa
        else {}
    )
    dataset = simulate_central_transition(
        cq_hz=cq_hz,
        eta=eta,
        larmor_hz=larmor_hz,
        spectral_width_hz=80e3,
        spinning_hz=spinning_hz,
        sidebands=2,
        spin=spin,
        **csa,
    )

    # Sample rotor axes over the whole sphere in the tensor frame. The
    # field sweeps a cone around each rotor axis, parametrized by two
    # orthogonal tangent vectors in the right-handed rotor frame.
    cosine, weights = np.polynomial.legendre.leggauss(24)
    z, azimuth = np.meshgrid(cosine, 2 * np.pi * np.arange(48) / 48, indexing="ij")
    sine = np.sqrt(1 - z**2)
    cos_phi, sin_phi = np.cos(azimuth), np.sin(azimuth)
    rotor_axis = np.stack((sine * cos_phi, sine * sin_phi, z), axis=-1)
    polar = np.stack((z * cos_phi, z * sin_phi, -sine), axis=-1)
    tangential = np.stack((-sin_phi, cos_phi, np.zeros_like(z)), axis=-1)
    rotor_points = 128
    angles = 2 * np.pi * np.arange(rotor_points) / rotor_points
    tilt = np.deg2rad(54.736)
    directions = np.cos(tilt) * rotor_axis[..., None, :] + np.sin(tilt) * (
        np.cos(angles)[:, None] * polar[..., None, :]
        + np.sin(angles)[:, None] * tangential[..., None, :]
    )
    shifts = _second_order_offsets(directions, cq_hz, eta, larmor_hz, spin)
    if with_csa:
        field_in_csa_axes = directions @ rotation
        shifts += field_in_csa_axes**2 @ np.array([-840, -360, 1200])
    modulation = shifts - shifts.mean(axis=-1, keepdims=True)

    # Integrate the cyclic modulation by trapezoids and project the
    # complex signal onto each rotor order by direct summation.
    steps = (modulation + np.roll(modulation, 1, axis=-1)) / 2
    phase = 2 * np.pi * steps.cumsum(axis=-1) / (rotor_points * spinning_hz)
    orders = np.arange(-2, 3)
    basis = np.exp(-1j * angles[:, None] * orders)
    coefficients = np.exp(1j * phase) @ basis / rotor_points
    areas = weights @ (np.abs(coefficients) ** 2).mean(axis=1) / 2
    expected = areas / areas.sum()

    # For both spins the mean offsets lie between -5.3 kHz and zero.
    # These boundaries separate the five bands without cutting their support.
    windows = [-np.inf, -17.5e3, -7.5e3, 2.5e3, 12.5e3, np.inf]
    actual, _ = np.histogram(dataset.x.data, bins=windows, weights=dataset.data)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=8e-5)


@pytest.mark.parametrize("spin", [0.5, 2, 2.25])
def test_central_transition_rejects_unsupported_spin(spin):
    with pytest.raises(ValueError, match="spin must be a half-integer"):
        simulate_central_transition(
            cq_hz=1e6, eta=0, larmor_hz=100e6, spectral_width_hz=60e3, spin=spin
        )


@pytest.mark.parametrize("anisotropy", [-5000, 5000])
def test_static_csa_matches_tensor_moments(anisotropy):
    eta = 0.6
    dataset = simulate_csa(anisotropy_hz=anisotropy, eta=eta, spectral_width_hz=16000)
    axis = dataset.x.data
    centers = axis + (axis[-1] - axis[0]) / (2 * (axis.size - 1))
    mean = dataset.data @ centers
    variance = dataset.data @ ((centers - mean) ** 2)
    # For an isotropic unit vector, E[n_i**4]=1/5 and E[n_i**2*n_j**2]=1/15.
    expected_std = abs(anisotropy) * np.sqrt((1 + eta**2 / 3) / 5)
    assert mean == pytest.approx(0, abs=1)
    assert np.sqrt(variance) == pytest.approx(expected_std, abs=2)
    positive = dataset.data > 0
    eigenvalues = anisotropy * np.array([-(1 + eta) / 2, -(1 - eta) / 2, 1])
    assert centers[positive].min() == pytest.approx(eigenvalues.min(), abs=2)
    assert centers[positive].max() == pytest.approx(eigenvalues.max(), abs=2)


@pytest.mark.parametrize(("tilt_deg", "spinning"), [(54.736, 2000), (30, 10000)])
def test_mas_csa_areas_match_cartesian_time_domain_reference(tilt_deg, spinning):
    anisotropy, eta = 5000, 0.6
    dataset = simulate_csa(
        anisotropy_hz=anisotropy,
        eta=eta,
        spectral_width_hz=8 * spinning,
        spinning_hz=spinning,
        rotor_angle_deg=tilt_deg,
        sidebands=2,
    )

    cosine, weights = np.polynomial.legendre.leggauss(24)
    z, phi = np.meshgrid(cosine, np.arange(48) * 2 * np.pi / 48, indexing="ij")
    sine = np.sqrt(1 - z**2)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    rotor = np.stack((sine * cos_phi, sine * sin_phi, z), axis=-1)
    polar = np.stack((z * cos_phi, z * sin_phi, -sine), axis=-1)
    azimuthal = np.stack((-sin_phi, cos_phi, np.zeros_like(z)), axis=-1)
    angles = np.arange(256) * 2 * np.pi / 256
    tilt = np.deg2rad(tilt_deg)
    field = np.cos(tilt) * rotor[..., None, :] + np.sin(tilt) * (
        np.cos(angles)[:, None] * polar[..., None, :]
        + np.sin(angles)[:, None] * azimuthal[..., None, :]
    )
    eigenvalues = anisotropy * np.array([-(1 + eta) / 2, -(1 - eta) / 2, 1])
    shift = field**2 @ eigenvalues
    modulation = shift - shift.mean(axis=-1, keepdims=True)
    steps = (modulation + np.roll(modulation, 1, axis=-1)) / 2
    phase = 2 * np.pi * steps.cumsum(axis=-1) / (angles.size * spinning)
    orders = np.arange(-2, 3)
    coefficients = np.exp(1j * phase) @ np.exp(-1j * angles[:, None] * orders)
    areas = weights @ (np.abs(coefficients) ** 2).mean(axis=1)
    expected = areas / areas.sum()

    boundaries = np.r_[-np.inf, np.arange(-1.5, 2) * spinning, np.inf]
    actual, _ = np.histogram(dataset.x.data, bins=boundaries, weights=dataset.data)
    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=0)


def test_rotated_csa_rejects_reflection():
    with pytest.raises(ValueError, match="proper orthogonal matrix"):
        simulate_central_transition(
            cq_hz=1e6,
            eta=0,
            larmor_hz=100e6,
            spectral_width_hz=20e3,
            csa_anisotropy_hz=1000,
            csa_rotation=np.diag([-1, 1, 1]),
        )


def test_aligned_csa_matches_explicit_identity_rotation():
    settings = dict(
        cq_hz=2e6,
        eta=0.8,
        larmor_hz=100e6,
        spectral_width_hz=30e3,
        csa_anisotropy_hz=-3000,
        csa_eta=0.6,
        spinning_hz=5e3,
        sidebands=2,
    )
    implicit = simulate_central_transition(**settings)
    explicit = simulate_central_transition(**settings, csa_rotation=np.eye(3))
    np.testing.assert_allclose(implicit.data, explicit.data, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    ("spin", "transitions"), [(1, "all"), (1.5, "satellites"), (2.5, "all")]
)
def test_static_quadrupolar_multiplet_matches_spin_hamiltonian(spin, transitions):
    cq_hz, eta, larmor_hz = 1e6, 0.65, 100e6
    dataset = simulate_quadrupolar(
        cq_hz=cq_hz,
        eta=eta,
        larmor_hz=larmor_hz,
        spectral_width_hz=2e6,
        points=32768,
        accuracy=32,
        spin=spin,
        transitions=transitions,
    )
    cosine, weights = np.polynomial.legendre.leggauss(32)
    z, phi = np.meshgrid(cosine, 2 * np.pi * np.arange(64) / 64, indexing="ij")
    sine = np.sqrt(1 - z**2)
    directions = np.stack((sine * np.cos(phi), sine * np.sin(phi), z), axis=-1)
    first, second = _perturbative_level_offsets(directions, cq_hz, eta, larmor_hz, spin)
    offsets = np.diff(first + second, axis=-1)
    # The Zeeman eigenstates are ordered from m=+I to m=-I.
    m = np.arange(spin, -spin, -1)
    strengths = spin * (spin + 1) - m * (m - 1)
    if transitions == "satellites":
        strengths[m == 0.5] = 0
    strengths = strengths / strengths.sum()
    ref_mean = weights @ (offsets @ strengths).mean(axis=1) / 2
    ref_std = np.sqrt(
        weights @ (((offsets - ref_mean) ** 2) @ strengths).mean(axis=1) / 2
    )

    frequency = dataset.x.data
    centers = frequency + (frequency[1] - frequency[0]) / 2
    profile = dataset.data
    mean = profile @ centers
    assert profile.sum() == pytest.approx(1)
    assert mean == pytest.approx(ref_mean, abs=8)
    assert np.sqrt(profile @ (centers - mean) ** 2) == pytest.approx(ref_std, rel=4e-4)


def test_quadrupolar_multiplet_rejects_unbounded_spin():
    with pytest.raises(ValueError, match="multiplets support spin up to 16"):
        simulate_quadrupolar(
            cq_hz=1e6, eta=0, larmor_hz=100e6, spectral_width_hz=1e6, spin=1e9
        )


@pytest.mark.parametrize("offset", [-1250.0, 0.0, 1250.0])
def test_static_rectangular_pulse_matches_bloch_rotation(offset):
    rf_hz, duration = 2500, 73e-6
    result = simulate_rectangular_pulse(
        spin=0.5,
        larmor_hz=100e6,
        spectral_width_hz=5000,
        rf_hz=rf_hz,
        duration_s=duration,
        shift_hz=offset,
        points=1001,
        accuracy=2,
    )
    effective_rf = np.hypot(rf_hz, offset)
    angle = 2 * np.pi * effective_rf * duration
    expected = rf_hz / effective_rf * np.sin(angle) + (
        1j * rf_hz * offset / effective_rf**2 * (1 - np.cos(angle))
    )
    assert result.data.sum() == pytest.approx(expected, abs=1e-12)
    assert result.x.data[np.argmax(np.abs(result.data))] == pytest.approx(offset)


def test_hard_rectangular_pulse_approaches_ideal_quadrupolar_powder():
    settings = dict(
        spin=1.5,
        cq_hz=600e3,
        eta=0.6,
        larmor_hz=100e6,
        spectral_width_hz=1e6,
        points=2048,
        accuracy=6,
    )
    rf_hz = 1e9
    finite = simulate_rectangular_pulse(
        **settings, rf_hz=rf_hz, duration_s=0.25 / rf_hz
    )
    ideal = simulate_quadrupolar(**settings)
    assert np.abs(finite.data.real - ideal.data).sum() < 1e-6
    assert np.abs(finite.data.imag).sum() < 2e-4


def test_hard_rectangular_mas_pulse_approaches_ideal_csa_bands():
    settings = dict(
        spectral_width_hz=20e3,
        spinning_hz=2000,
        sidebands=2,
        points=4096,
        accuracy=4,
    )
    rf_hz = 1e9
    finite = simulate_rectangular_pulse(
        spin=0.5,
        larmor_hz=100e6,
        csa_anisotropy_hz=5000,
        csa_eta=0.6,
        rf_hz=rf_hz,
        duration_s=0.25 / rf_hz,
        **settings,
    )
    ideal = simulate_csa(anisotropy_hz=5000, eta=0.6, **settings)
    # The ideal profile is normalized over retained bands. The RF result
    # instead keeps the intensity lost into omitted orders.
    real = finite.data.real
    assert 0 < real.sum() < 1
    np.testing.assert_allclose(real / real.sum(), ideal.data, rtol=0, atol=2e-8)
    assert np.abs(finite.data.imag).sum() < 1e-5


def test_finite_mas_pulse_converges_when_time_step_is_halved():
    settings = dict(
        spin=0.5,
        larmor_hz=100e6,
        csa_anisotropy_hz=5000,
        csa_eta=0.6,
        spectral_width_hz=20e3,
        spinning_hz=5000,
        sidebands=1,
        points=4096,
        accuracy=2,
        rf_hz=2000,
        duration_s=100e-6,
    )
    coarse, fine, reference = (
        simulate_rectangular_pulse(**settings, max_rf_step_s=step).data
        for step in (4e-6, 2e-6, 0.5e-6)
    )
    assert np.abs(fine - reference).sum() < np.abs(coarse - reference).sum() / 3
    assert np.abs(fine - reference).sum() < 5e-4


def test_finite_mas_pulse_uses_hemisphere_for_aligned_tensors(monkeypatch):
    from spectrochempy_nmr import pulsar

    settings = dict(
        spin=0.5,
        csa_anisotropy_hz=11000,
        csa_eta=0.45,
        rf_hz=4800,
        duration_s=61e-6,
        shift_hz=641,
        spinning_hz=6500,
        spectral_width_hz=50e3,
        larmor_hz=100e6,
        sidebands=3,
        accuracy=2,
    )
    hemisphere = simulate_rectangular_pulse(**settings)
    original_mesh = pulsar._powder_mesh

    def full_sphere(accuracy, hemisphere=False, full_sphere=False):
        directions, weights, triangles = original_mesh(accuracy, True)
        return (
            np.concatenate((directions, -directions), axis=1),
            np.tile(weights, 2),
            np.concatenate((triangles, triangles + weights.size)),
        )

    monkeypatch.setattr(pulsar, "_powder_mesh", full_sphere)
    full = simulate_rectangular_pulse(**settings)
    np.testing.assert_allclose(hemisphere.data, full.data, atol=1e-12, rtol=0)


def test_finite_mas_pulse_rejects_oversized_single_vertex_matrix_batch():
    with pytest.raises(ValueError, match="matrix batch limit"):
        simulate_rectangular_pulse(
            spin=16,
            larmor_hz=100e6,
            spectral_width_hz=1e6,
            spinning_hz=1000,
            rf_hz=2000,
            duration_s=50e-6,
            accuracy=2,
        )


def test_rotated_csa_finite_mas_pulse_uses_full_sphere(monkeypatch):
    from scipy.spatial.transform import Rotation

    from spectrochempy_nmr import pulsar

    rotation = Rotation.from_euler("zyx", [42, 31, 19], degrees=True).as_matrix()
    settings = dict(
        spin=1.5,
        cq_hz=21000,
        eta=0.7,
        csa_anisotropy_hz=11000,
        csa_eta=0.45,
        csa_rotation=rotation,
        rf_hz=4800,
        duration_s=133e-6,
        shift_hz=641,
        spinning_hz=6500,
        spectral_width_hz=100e3,
        larmor_hz=100e6,
        sidebands=3,
        accuracy=2,
    )
    actual = simulate_rectangular_pulse(**settings)
    original_mesh = pulsar._powder_mesh

    def independently_reflected_mesh(accuracy, hemisphere=False, full_sphere=False):
        directions, weights, triangles = original_mesh(accuracy, True)
        return (
            np.concatenate((directions, -directions), axis=1),
            np.tile(weights, 2),
            np.concatenate((triangles, triangles + weights.size)),
        )

    monkeypatch.setattr(pulsar, "_powder_mesh", independently_reflected_mesh)
    expected = simulate_rectangular_pulse(**settings)
    np.testing.assert_allclose(actual.data, expected.data, atol=1e-12, rtol=0)


@pytest.mark.parametrize("spinning", [0, 5000])
def test_hahn_echo_recovers_the_ideal_profile_with_t2_at_equal_delays(spinning):
    delay = 200e-6
    t2 = 1e-3
    settings = dict(
        spin=0.5,
        larmor_hz=100e6,
        spectral_width_hz=30e3,
        shift_hz=800,
        csa_anisotropy_hz=2100,
        csa_eta=0.6,
        spinning_hz=spinning,
        sidebands=bool(spinning),
        points=4096,
        accuracy=3,
    )
    ideal = simulate_pulse_sequence([IdealPulse(90)], **settings)
    echo = simulate_pulse_sequence(
        [IdealPulse(90), Delay(delay), IdealPulse(180), Delay(delay)],
        t2_s=t2,
        receiver_phase_deg=180,
        **settings,
    )
    np.testing.assert_allclose(
        echo.data, ideal.data * np.exp(-2 * delay / t2), atol=1e-12
    )


@pytest.mark.parametrize(("order", "expected"), [(1, 1), (-1, 0)])
def test_sequence_coherence_selection_uses_spin_projection_difference(order, expected):
    result = simulate_pulse_sequence(
        [IdealPulse(90), Coherence(order)],
        spin=0.5,
        larmor_hz=100e6,
        spectral_width_hz=30e3,
        accuracy=2,
    )
    assert result.data.sum() == pytest.approx(expected, abs=1e-12)


def test_offset_pulse_returns_coherence_to_the_simulation_carrier():
    offset, duration = 1250, 100e-6
    result = simulate_pulse_sequence(
        [Pulse(duration, 0.25 / duration, offset_hz=offset)],
        spin=0.5,
        larmor_hz=100e6,
        spectral_width_hz=5000,
        shift_hz=offset,
        accuracy=2,
    )
    expected = np.exp(2j * np.pi * offset * duration)
    assert result.data.sum() == pytest.approx(expected, abs=1e-12)


def test_rf_central_selection_allows_a_narrow_window():
    settings = dict(
        cq_hz=500e3,
        eta=0.6,
        spin=1.5,
        larmor_hz=100e6,
        spectral_width_hz=20e3,
        accuracy=4,
    )
    central = simulate_pulse_sequence(
        [IdealPulse(90)], transitions="central", **settings
    )
    ideal = simulate_central_transition(**settings)
    np.testing.assert_allclose(central.data, ideal.data, atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match="spectral window"):
        simulate_pulse_sequence([IdealPulse(90)], **settings)


def test_two_equivalent_passive_spins_give_one_two_one_triplet():
    settings = dict(
        spectral_width_hz=2000,
        points=2001,
        accuracy=2,
        j_partners=(PassiveJ(0.5, 200), PassiveJ(0.5, 200)),
    )
    ideal = simulate_csa(anisotropy_hz=0, eta=0, **settings)
    rf = simulate_pulse_sequence(
        [IdealPulse(90)], spin=0.5, larmor_hz=100e6, **settings
    )
    selected = np.flatnonzero(ideal.data)
    np.testing.assert_array_equal(ideal.x.data[selected], [-200, 0, 200])
    np.testing.assert_allclose(ideal.data[selected], [0.25, 0.5, 0.25])
    np.testing.assert_allclose(rf.data, ideal.data, atol=1e-12, rtol=0)


def test_passive_j_offset_also_acts_during_preparation():
    j_hz, delay = 400, 0.5e-3
    result = simulate_pulse_sequence(
        [IdealPulse(90), Delay(delay)],
        spin=0.5,
        larmor_hz=100e6,
        spectral_width_hz=2000,
        points=2001,
        accuracy=2,
        j_partners=[PassiveJ(0.5, j_hz)],
    )
    peaks = np.flatnonzero(np.abs(result.data) > 1e-10)
    np.testing.assert_array_equal(result.x.data[peaks], [-j_hz / 2, j_hz / 2])
    expected = np.exp(2j * np.pi * result.x.data[peaks] * delay) / 2
    np.testing.assert_allclose(result.data[peaks], expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("spinning", [0, 6500])
def test_coupled_sequence_matches_passive_limit_for_an_unpulsed_partner(spinning):
    events = [Pulse(25e-6, 7500, phase_deg=15), Delay(62e-6)]
    settings = dict(
        spectral_width_hz=50e3,
        points=4096,
        accuracy=2,
        spinning_hz=spinning,
        sidebands=bool(spinning),
    )
    site = dict(spin=1.5, cq_hz=12000, eta=0.5, larmor_hz=100e6)
    coupled = simulate_coupled_sequence(
        events,
        observed=SpinSite(**site),
        partner=SpinSite(0.5, 80e6),
        j_hz=300,
        **settings,
    )
    passive = simulate_pulse_sequence(
        events,
        j_partners=(PassiveJ(0.5, 300),),
        **site,
        **settings,
    )
    np.testing.assert_allclose(coupled.data, passive.data, atol=1e-12, rtol=0)


@pytest.mark.parametrize("refocus", ["partner_pi", "decoupled_delay"])
def test_active_partner_refocuses_scalar_coupling(refocus):
    settings = dict(
        observed=SpinSite(0.5, 100e6),
        partner=SpinSite(0.5, 80e6),
        j_hz=500,
        spectral_width_hz=2000,
        points=2001,
        accuracy=2,
    )
    reference = simulate_coupled_sequence([IdealPulse(90)], **settings)
    if refocus == "partner_pi":
        events = [
            IdealPulse(90),
            Delay(0.001),
            IdealPulse(0, partner_flip_deg=180),
            Delay(0.001),
        ]
    else:
        events = [IdealPulse(90), Delay(0.002, decouple_j=True)]
    actual = simulate_coupled_sequence(events, **settings)
    np.testing.assert_allclose(actual.data, reference.data, atol=1e-12, rtol=0)


def test_rotated_site_harmonics_use_the_common_rotor_phase():
    from scipy.spatial.transform import Rotation

    from spectrochempy_nmr.pulsar import _site_parameters, _site_transitions

    rotation = Rotation.from_euler("zyx", [42, 31, 19], degrees=True).as_matrix()
    n = np.array([0.36, 0.48, 0.8])
    polar = np.array([0.48, 0.64, -0.6])
    azimuthal = np.array([-0.8, 0.6, 0])
    tilt = np.deg2rad(54.736)
    phase = np.arange(64) * 2 * np.pi / 64
    trajectory = np.cos(tilt) * n + np.sin(tilt) * (
        np.cos(phase)[:, None] * polar + np.sin(phase)[:, None] * azimuthal
    )
    site = SpinSite(1.5, 100e6, cq_hz=500e3, eta=0.7, efg_rotation=rotation)
    terms, _ = _site_transitions(n[:, None], tilt, _site_parameters(site), 5)
    reconstructed = (terms[0] @ np.exp(-1j * np.arange(5)[:, None] * phase)).real
    first, second = _perturbative_level_offsets(
        trajectory @ rotation, 500e3, 0.7, 100e6, 1.5
    )
    np.testing.assert_allclose(
        reconstructed, np.diff(first + second, axis=-1).T, atol=1e-8, rtol=0
    )


def test_active_coupling_workspace_counts_both_site_tables():
    with pytest.raises(ValueError, match="Coupled-spin work arrays exceed"):
        simulate_coupled_sequence(
            [IdealPulse(90)],
            observed=SpinSite(0.5, 100e6, efg_rotation=np.eye(3)),
            partner=SpinSite(0.5, 80e6),
            j_hz=200,
            spectral_width_hz=20e3,
            spinning_hz=10e3,
            accuracy=28,
        )


def test_aligned_coupled_mas_half_turn_symmetry_with_odd_requested_grid(monkeypatch):
    from spectrochempy_nmr import pulsar

    events = [
        Pulse(77e-6, 4800, 17, 1000, 3600, 42, 620),
        Delay(62e-6),
        Pulse(33e-6, 7000, -40, -940, 2200, 56, -700),
    ]
    settings = dict(
        observed=SpinSite(
            1.5, 100e6, cq_hz=2000, eta=0.7, csa_anisotropy_hz=11000, csa_eta=0.43
        ),
        partner=SpinSite(0.5, 80e6, csa_anisotropy_hz=14000, csa_eta=0.5),
        j_hz=850,
        spectral_width_hz=39000,
        spinning_hz=6500,
        sidebands=1,
        accuracy=1,
    )
    hemisphere = simulate_coupled_sequence(events, **settings)
    original_mesh = pulsar._powder_mesh

    def full_mesh(accuracy, hemisphere=False, full_sphere=False):
        return original_mesh(accuracy, True, True)

    monkeypatch.setattr(pulsar, "_powder_mesh", full_mesh)
    full = simulate_coupled_sequence(events, **settings)
    np.testing.assert_allclose(hemisphere.data, full.data, atol=1e-12, rtol=0)
