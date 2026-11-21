"""Installed sequence templates and their complex simulator parity."""

from pathlib import Path

import numpy as np
import pytest
from spectrochempy.nmr import pulsar


PARAMETERS = {
    "p1": "3 us",
    "p2": "2 us",
    "p3": "8 us",
    "rf1": "100 kHz",
    "rf2": "70 kHz",
    "ph1": 0,
    "ph2": 90,
    "ph3": 0,
    "rf_exc": "100 kHz",
    "rf_conv": "80 kHz",
    "rf_read": "10 kHz",
    "q": -3,
    "d0": "12.5 us",
    "d1": "50 us",
    "d2": "50 us",
    "dz": "100 us",
    "l1": 1,
    "o1": "1 kHz",
    "o2": "-800 Hz",
}


def test_installed_library_catalog_and_hahn_companion():
    assert pulsar.available_sequences() == (
        "hahn",
        "ideal",
        "mqmas",
        "mqmas_zfilter",
        "rectangular",
        "two_channel",
    )
    companion = Path(__file__).resolve().parents[1] / "examples/simulation/hahn.seq"
    external = pulsar.read_sequence(companion).build(PARAMETERS)
    assert pulsar.load_sequence("hahn").build(PARAMETERS) == external


@pytest.mark.parametrize(
    "name", ["hahn", "mqmas", "mqmas_zfilter", "rectangular", "two_channel"]
)
def test_every_library_program_binds_and_can_be_reused(name):
    program = pulsar.load_sequence(name)
    first = program.build(PARAMETERS)
    second = program.build({**PARAMETERS, "p1": "6 us"})
    assert first[0].duration_s == pytest.approx(3e-6)
    assert second[0].duration_s == pytest.approx(6e-6)
    assert program.build(PARAMETERS) == first
    assert program.source == f"library:{name}"


@pytest.mark.parametrize("name", ["../hahn", "unknown", "hahn.seq", ""])
def test_library_lookup_does_not_resolve_external_paths(name):
    with pytest.raises(ValueError, match="Unknown sequence"):
        pulsar.load_sequence(name)


def test_library_errors_identify_template_and_line():
    with pytest.raises(pulsar.SequenceError) as caught:
        pulsar.load_sequence("rectangular").build({"rf1": "1 kHz"})
    assert caught.value.source == "library:rectangular"
    assert caught.value.line == 3
    with pytest.raises(TypeError):
        pulsar.load_sequence(None)


@pytest.mark.parametrize("order", [3, -3])
def test_library_mqmas_matches_manual_complex_propagation(order):
    events = pulsar.load_sequence("mqmas").build({**PARAMETERS, "q": order})
    manual = (
        pulsar.Pulse(3e-6, 100e3),
        pulsar.Coherence(order),
        pulsar.Delay(12.5e-6),
        pulsar.Pulse(2e-6, 80e3, 90),
        pulsar.Coherence(1),
    )
    settings = dict(
        spin=1.5,
        larmor_hz=105.804e6,
        cq_hz=1.5e6,
        spectral_width_hz=20000,
        points=1024,
        accuracy=3,
        transitions="central",
        receiver_phase_deg=37,
    )
    actual = pulsar.simulate_pulse_sequence(events, **settings)
    expected = pulsar.simulate_pulse_sequence(manual, **settings)
    assert np.max(np.abs(expected.data)) > 1e-9
    np.testing.assert_allclose(actual.data, expected.data, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("duration", [0.5, 3])
def test_library_rectangular_matches_manual_complex_mas(duration):
    events = pulsar.load_sequence("rectangular").build(
        {**PARAMETERS, "p1": f"{duration} us"}
    )
    settings = dict(
        spin=2.5,
        larmor_hz=104.228e6,
        cq_hz=2.5e6,
        eta=0.3,
        spinning_hz=5000,
        spectral_width_hz=30000,
        sidebands=2,
        points=1024,
        accuracy=2,
        transitions="central",
    )
    actual = pulsar.simulate_pulse_sequence(events, **settings)
    expected = pulsar.simulate_pulse_sequence(
        [pulsar.Pulse(duration * 1e-6, 100e3)], **settings
    )
    np.testing.assert_allclose(actual.data, expected.data, rtol=1e-10, atol=1e-12)
