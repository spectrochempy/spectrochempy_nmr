"""Receiver weighting, complex areas and archive compatibility."""

import json

import numpy as np
import pytest

from spectrochempy_nmr import CzjzekDistribution
from spectrochempy_nmr import CzjzekRFLibrary
from spectrochempy_nmr import fit_czjzek
from spectrochempy_nmr import simulate_czjzek
from spectrochempy_nmr.pulsar import Pulse
from spectrochempy_nmr.pulsar import simulate_pulse_sequence

EVENTS = (Pulse(1e-6, 50e3, 25),)
SETTINGS = dict(
    spin=2.5,
    larmor_hz=195e6,
    spectral_width_hz=16e3,
    points=129,
    accuracy=1,
    spinning_hz=2e3,
    sidebands=2,
    transitions="central",
)
GRID = dict(cq_grid_hz=[1.8e6, 2e6], eta_grid=[0.2, 0.5], shift_hz=300, shift_bins=2)


def test_finite_receiver_matches_native_member_spectra_and_complex_band_areas():
    distribution = CzjzekDistribution(100e3, 2e6, 0.5, 300, 20)
    members = distribution.sample(4, seed=17)
    received = simulate_czjzek(
        distribution,
        events=EVENTS,
        samples=4,
        seed=17,
        probe_bandwidth_hz=8e3,
        **SETTINGS,
    )
    expected = np.zeros(SETTINGS["points"], complex)
    for cq, eta, shift, weight in zip(
        members.cq_hz, members.eta, members.shift_hz, members.weights, strict=True
    ):
        native = simulate_pulse_sequence(
            EVENTS,
            cq_hz=cq,
            eta=eta,
            shift_hz=shift,
            probe_bandwidth_hz=8e3,
            **SETTINGS,
        )
        expected += weight * native.data
    np.testing.assert_allclose(received.data, expected, rtol=1e-13, atol=1e-15)
    for order in range(-2, 3):
        window = abs(received.x.data - order * SETTINGS["spinning_hz"]) < 750
        np.testing.assert_allclose(
            received.data[window].sum(), expected[window].sum(), atol=1e-15
        )
    flat = simulate_czjzek(distribution, events=EVENTS, samples=4, seed=17, **SETTINGS)
    explicit_flat = simulate_czjzek(
        distribution,
        events=EVENTS,
        samples=4,
        seed=17,
        probe_bandwidth_hz=None,
        **SETTINGS,
    )
    np.testing.assert_array_equal(flat.data, explicit_flat.data)
    assert np.linalg.norm(received.data - flat.data) > 1e-5
    assert abs(received.data.sum()) < abs(flat.data.sum())


def test_receiver_interpolation_with_interior_shift_disorder():
    settings = {**SETTINGS, "points": 1025, "accuracy": 2, "probe_bandwidth_hz": 8e3}
    grid = {**GRID, "shift_bins": 8}
    library = CzjzekRFLibrary.build(EVENTS, **grid, **settings)
    distribution = CzjzekDistribution(0, 2e6, 0.5, 300, 25)
    actual = library.simulate(distribution, samples=4, seed=17).data
    expected = simulate_czjzek(
        distribution, events=EVENTS, samples=4, seed=17, **settings
    ).data
    # Off-plane interpolation is approximate, including deposition and RF
    # excitation; retain an independently calculated direct control.
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.025
    axis = np.linspace(-8e3, 8e3, settings["points"])
    areas = []
    for spectrum in (actual, expected):
        mass = np.array(
            [
                spectrum[abs(axis - order * settings["spinning_hz"]) < 750].sum()
                for order in range(-2, 3)
            ]
        )
        areas.append(100 * mass / mass[2])
    np.testing.assert_allclose(areas[0], areas[1], rtol=0, atol=0.005)


@pytest.mark.parametrize("bandwidth", [None, 8e3])
def test_receiver_archive_roundtrip_and_fit_mismatch(tmp_path, bandwidth):
    settings = {**SETTINGS, "probe_bandwidth_hz": bandwidth}
    library = CzjzekRFLibrary.build(EVENTS, **GRID, **settings)
    filename = tmp_path / "library.npz"
    library.save(filename)
    with np.load(filename, allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].item())
    if bandwidth is None:
        # Same format-1 settings as archives made before receiver support.
        assert "probe_bandwidth_hz" not in metadata["settings"]
        loaded = CzjzekRFLibrary.load(filename, EVENTS, **GRID, **SETTINGS)
        library._validate_fit_model(EVENTS, SETTINGS)
    else:
        assert metadata["settings"]["probe_bandwidth_hz"] == bandwidth
        loaded = CzjzekRFLibrary.load(filename, EVENTS, **GRID, **settings)
    step = SETTINGS["spectral_width_hz"] / (SETTINGS["points"] - 1)
    target = CzjzekDistribution(0, 2e6, 0.5, 300 + 2 * step)
    np.testing.assert_array_equal(
        loaded.simulate(target).data, library.simulate(target).data
    )
    other_widths = (8e3, 16e3) if bandwidth is None else (None, 16e3)
    for other in other_widths:
        mismatch = {**SETTINGS, "probe_bandwidth_hz": other}
        with pytest.raises(ValueError, match="does not match"):
            CzjzekRFLibrary.load(filename, EVENTS, **GRID, **mismatch)
        with pytest.raises(ValueError, match="does not match"):
            fit_czjzek(
                lambda *_: pytest.fail("Must reject before optimizing"),
                [0],
                [(0, 1)],
                events=EVENTS,
                library=library,
                **mismatch,
            )


@pytest.mark.parametrize("bad", [0, -1, np.inf, np.nan, 8e3 + 1j, "8000"])
def test_invalid_receiver_rejected_for_direct_and_archived_models(tmp_path, bad):
    settings = {**SETTINGS, "probe_bandwidth_hz": bad}
    with pytest.raises((TypeError, ValueError)):
        simulate_czjzek(CzjzekDistribution(0, 2e6), **settings)
    with pytest.raises((TypeError, ValueError)):
        CzjzekRFLibrary.build(EVENTS, **GRID, **settings)
    with pytest.raises((TypeError, ValueError)):
        CzjzekRFLibrary.load(tmp_path / "unused.npz", EVENTS, **GRID, **settings)
