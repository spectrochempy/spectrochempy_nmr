"""Regression checkpoints for the NumPy port of Masaï PULSAR."""

import numpy as np
import pytest

from spectrochempy.core.units import ur
from spectrochempy_nmr.pulsar import simulate_central_transition
from spectrochempy_nmr.pulsar import simulate_csa


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


def _second_order_offsets(directions, cq_hz, eta, larmor_hz, spin):
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
    center = levels // 2
    return corrections[..., center] - corrections[..., center - 1]


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


@pytest.mark.parametrize("spin", [1.5, 2.5])
def test_mas_sideband_areas_match_time_domain_spin_hamiltonian(spin):
    cq_hz, larmor_hz, eta, spinning_hz = 2.8e6, 100e6, 0.7, 10e3
    dataset = simulate_central_transition(
        cq_hz=cq_hz,
        eta=eta,
        larmor_hz=larmor_hz,
        spectral_width_hz=80e3,
        spinning_hz=spinning_hz,
        sidebands=2,
        spin=spin,
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


def test_mas_csa_areas_match_cartesian_time_domain_reference():
    anisotropy, eta, spinning = 5000, 0.6, 2000
    dataset = simulate_csa(
        anisotropy_hz=anisotropy,
        eta=eta,
        spectral_width_hz=16000,
        spinning_hz=spinning,
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
    tilt = np.deg2rad(54.736)
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

    boundaries = [-np.inf, -3000, -1000, 1000, 3000, np.inf]
    actual, _ = np.histogram(dataset.x.data, bins=boundaries, weights=dataset.data)
    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=0)
