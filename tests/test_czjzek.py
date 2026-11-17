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


def test_rf_library_exact_nodes_roundtrip_and_model_validation(tmp_path):
    from fractions import Fraction

    from spectrochempy_nmr import CzjzekRFLibrary
    from spectrochempy_nmr.pulsar import Delay
    from spectrochempy_nmr.pulsar import Pulse

    events = (Pulse(7e-6, 24e3, 23), Delay(13e-6))
    settings = dict(
        spin=2.5,
        larmor_hz=104.468516e6,
        spectral_width_hz=48e3,
        spinning_hz=8e3,
        points=513,
        accuracy=1,
        sidebands=2,
    )
    grid = dict(
        cq_grid_hz=[2.8e6, 3e6], eta_grid=[0.2, 0.5], shift_hz=325, shift_bins=3
    )
    library = CzjzekRFLibrary.build(
        events,
        **grid,
        **{**settings, "spin": Fraction(5, 2), "larmor_hz": np.longdouble("104468516")},
    )
    step = settings["spectral_width_hz"] / (settings["points"] - 1)
    for offset in (-3, 3):
        target = CzjzekDistribution(0, 3e6, 0.5, 325 + offset * step)
        expected = simulate_czjzek(target, events=events, **settings)
        actual = library.simulate(target)
        np.testing.assert_array_equal(actual.x.data, expected.x.data)
        np.testing.assert_allclose(actual.data, expected.data, rtol=1e-12, atol=1e-14)

    target = CzjzekDistribution(0, 2.9e6, 0.35, 368)
    filename = tmp_path / "library.npz"
    library.save(filename)
    equivalent_events = (Pulse(7e-6, 24e3, np.int64(23), offset_hz=-0.0), events[1])
    loaded = CzjzekRFLibrary.load(filename, equivalent_events, **grid, **settings)
    np.testing.assert_array_equal(
        loaded.simulate(target).data, library.simulate(target).data
    )
    with pytest.raises(ValueError, match="does not match"):
        CzjzekRFLibrary.load(
            filename, events, **grid, **{**settings, "spinning_hz": 9000}
        )
    with pytest.raises(ValueError, match="does not match"):
        CzjzekRFLibrary.load(filename, (Pulse(7e-6, 24e3, 20),), **grid, **settings)

    invalid_cycle = PhaseCycle((events,), ("1",))
    with pytest.raises(ValueError, match="Supply one numeric coefficient"):
        CzjzekRFLibrary.load(filename, invalid_cycle, **grid, **settings)


def test_rf_library_uses_direct_response_outside_supported_region():
    from spectrochempy_nmr import CzjzekRFLibrary
    from spectrochempy_nmr.pulsar import Pulse

    events = (Pulse(7e-6, 24e3, 23),)
    settings = dict(
        spin=2.5,
        larmor_hz=104.468516e6,
        spectral_width_hz=np.float32(48001),
        spinning_hz=8e3,
        points=513,
        accuracy=1,
        sidebands=2,
    )
    library = CzjzekRFLibrary.build(
        events,
        cq_grid_hz=[2.8e6, 3e6],
        eta_grid=[0.2, 0.5],
        shift_hz=325,
        shift_bins=3,
        **settings,
    )
    targets = (
        CzjzekDistribution(0, -2.9e6, 0.35, 325),
        CzjzekDistribution(0, 2.9e6, 0.35, 800),
        CzjzekDistribution(30e3, 2.9e6, 0.35, 325, 35),
    )
    for target in targets:
        expected = simulate_czjzek(
            target, samples=8, seed=17, events=events, **settings
        )
        actual = library.simulate(target, samples=8, seed=17)
        np.testing.assert_array_equal(actual.x.data, expected.x.data)
        np.testing.assert_array_equal(actual.data, expected.data)


def test_rf_library_combines_interpolated_and_direct_members():
    from spectrochempy_nmr import CzjzekRFLibrary
    from spectrochempy_nmr.pulsar import Pulse

    events = (Pulse(7e-6, 24e3, 23),)
    settings = dict(
        spin=2.5,
        larmor_hz=104.468516e6,
        spectral_width_hz=48e3,
        spinning_hz=8e3,
        points=513,
        accuracy=1,
        sidebands=2,
    )
    target = CzjzekDistribution(85e3, 2.9e6, 0.35, 606.25)
    members = target.sample(2, seed=17)
    inside = np.argmax(members.cq_hz)
    cq, eta = members.cq_hz[inside], members.eta[inside]
    assert members.cq_hz[1 - inside] < cq
    library = CzjzekRFLibrary.build(
        events,
        cq_grid_hz=[cq, cq + 100e3],
        eta_grid=[0, eta],
        shift_hz=325,
        shift_bins=3,
        **settings,
    )
    expected = simulate_czjzek(target, samples=2, seed=17, events=events, **settings)
    actual = library.simulate(target, samples=2, seed=17)
    np.testing.assert_allclose(actual.data, expected.data, rtol=1e-12, atol=1e-14)
