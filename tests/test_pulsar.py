"""Regression checkpoints for the NumPy port of Masaï PULSAR."""

import numpy as np
import pytest

from spectrochempy.core.units import ur
from spectrochempy_nmr.pulsar import simulate_central_transition


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
