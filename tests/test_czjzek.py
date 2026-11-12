"""Independent EFG invariants and ensemble intensity conventions."""

import numpy as np
import pytest
from scipy.stats import chi

from spectrochempy_nmr import CzjzekDistribution
from spectrochempy_nmr import simulate_czjzek
from spectrochempy_nmr.czjzek import _BASIS
from spectrochempy_nmr.czjzek import _tensor_parameters
from spectrochempy_nmr.pulsar import IdealPulse
from spectrochempy_nmr.pulsar import PhaseCycle
from spectrochempy_nmr.pulsar import _central_transition_profile
from spectrochempy_nmr.pulsar import simulate_quadrupolar

SETTINGS = dict(
    spin=2.5,
    larmor_hz=130e6,
    spectral_width_hz=200e3,
    spinning_hz=12e3,
    points=2049,
    accuracy=2,
    sidebands=1,
)


def test_standard_radial_law_and_eta_marginal():
    nodes = CzjzekDistribution(1.0, shift_sigma_hz=2).sample(32768, seed=9)
    pq = np.abs(nodes.cq_hz) * np.sqrt(1 + nodes.eta**2 / 3)
    np.testing.assert_allclose(np.mean(pq**2), 10 / 3, rtol=0.002)
    np.testing.assert_allclose(
        np.quantile(pq, [0.1, 0.5, 0.9]),
        np.sqrt(2 / 3) * chi.ppf([0.1, 0.5, 0.9], 5),
        rtol=0.01,
    )
    # Integral of the universal eta density in the standard model.
    assert abs(nodes.eta.mean() - 0.6098231624487167) < 0.003
    assert abs(nodes.shift_hz.std() - 2) < 0.005
    assert abs(np.corrcoef(pq, nodes.shift_hz)[0, 1]) < 0.01
    assert np.all((nodes.eta >= 0) & (nodes.eta <= 1))
    np.testing.assert_allclose(nodes.weights.sum(), 1)
    assert not nodes.cq_hz.flags.writeable


def test_rotations_and_frobenius_identity():
    rng = np.random.default_rng(23)
    tensors = np.einsum("ni,ijk->njk", rng.normal(size=(100, 5)), _BASIS)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    cq, eta = _tensor_parameters(tensors)
    cq2, eta2 = _tensor_parameters(rotation @ tensors @ rotation.T)
    np.testing.assert_allclose(cq, cq2, atol=1e-14)
    np.testing.assert_allclose(eta, eta2, atol=1e-14)
    np.testing.assert_allclose(
        cq**2 * (1 + eta**2 / 3), 2 / 3 * np.sum(tensors**2, axis=(1, 2)), atol=1e-14
    )


def test_extended_second_moment_and_reproducibility():
    model = CzjzekDistribution(0.3, cq_hz=2, eta=0.4)
    a, b = model.sample(16384), model.sample(16384)
    np.testing.assert_array_equal(a.cq_hz, b.cq_hz)
    pq2 = a.cq_hz**2 * (1 + a.eta**2 / 3)
    np.testing.assert_allclose(
        pq2.mean(), 4 * (1 + 0.16 / 3) + 10 / 3 * 0.09, rtol=0.001
    )


def test_zero_disorder_matches_single_tensor():
    model = CzjzekDistribution(0, cq_hz=-2e6, eta=0.4, shift_hz=1000)
    nodes = model.sample(128)
    assert len(nodes.cq_hz) == 1
    expected = simulate_quadrupolar(
        cq_hz=-2e6, eta=0.4, shift_hz=1000, transitions="central", **SETTINGS
    )
    actual = simulate_czjzek(model, **SETTINGS)
    np.testing.assert_allclose(actual.data, expected.data)


def test_ideal_normalization_is_over_the_ensemble():
    model = CzjzekDistribution(0.9e6, cq_hz=3e6, eta=0.6)
    nodes = model.sample(8)
    raw = []
    normalized = []
    for cq, eta in zip(nodes.cq_hz, nodes.eta, strict=True):
        _, mass = _central_transition_profile(
            cq_hz=cq, eta=eta, **SETTINGS, _normalize=False
        )
        raw.append(mass)
        normalized.append(mass / mass.sum())
    expected = np.sum(raw, axis=0)
    expected /= expected.sum()
    actual = simulate_czjzek(model, samples=8, **SETTINGS)
    np.testing.assert_allclose(actual.data, expected, atol=1e-14)
    assert np.max(np.abs(actual.data - np.mean(normalized, axis=0))) > 1e-5


def test_finite_generator_preserves_excitation_scale():
    settings = {**SETTINGS, "accuracy": 1, "sidebands": 0}
    model = CzjzekDistribution(0.1e6)
    full = simulate_czjzek(model, samples=4, events=[IdealPulse(90)], **settings)
    small = simulate_czjzek(
        model, samples=4, events=(event for event in [IdealPulse(30)]), **settings
    )
    np.testing.assert_allclose(full.data, 2 * small.data, atol=1e-14)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(sigma_hz=-1),
        dict(sigma_hz=np.nan),
        dict(sigma_hz=0, eta=1.1),
        dict(sigma_hz=1, shift_sigma_hz=-1),
    ],
)
def test_invalid_model(kwargs):
    with pytest.raises(ValueError):
        CzjzekDistribution(**kwargs)


@pytest.mark.parametrize("size", [0, 3, 131072])
def test_invalid_sampling_size(size):
    with pytest.raises(ValueError):
        CzjzekDistribution(1).sample(size)


def test_phase_cycle_generators_replayed_for_every_tensor():
    settings = {**SETTINGS, "accuracy": 1, "sidebands": 0}
    model = CzjzekDistribution(0.1e6)

    def cycle():
        return PhaseCycle(
            ((pulse for pulse in [IdealPulse(angle)]) for angle in [90, -90]),
            (weight for weight in [0.5, -0.5]),
        )

    expected = simulate_czjzek(model, samples=4, events=[IdealPulse(90)], **settings)
    actual = simulate_czjzek(model, samples=4, events=cycle(), **settings)
    np.testing.assert_allclose(actual.data, expected.data, atol=1e-14)


def test_shift_disorder_preserves_local_tensor_at_eta_one():
    nodes = CzjzekDistribution(0, cq_hz=-2, eta=1, shift_sigma_hz=0.1).sample(8)
    np.testing.assert_array_equal(nodes.cq_hz, -2)
    np.testing.assert_array_equal(nodes.eta, 1)
