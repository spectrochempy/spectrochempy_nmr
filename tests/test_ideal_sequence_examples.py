"""Ideal-program physics and multisite superposition, outside the gallery."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import spectrochempy as scp
from spectrochempy.nmr import pulsar


@pytest.mark.parametrize("flip", [0, 45, 90, -90, 180])
def test_ideal_library_angle_and_absorption_scale(flip):
    events = pulsar.load_sequence("ideal").build({"a1": f"{flip} deg", "ph1": 0})
    spectrum = pulsar.simulate_pulse_sequence(
        events,
        spin=1,
        larmor_hz=57.792e6,
        spectral_width_hz=10000,
        accuracy=1,
        points=128,
    )
    np.testing.assert_allclose(spectrum.sum(), np.sin(np.deg2rad(flip)), atol=1e-12)


def test_ideal_angles_units_defaults_and_state():
    program = pulsar.parse_sequence(
        "set phase=30 deg\nideal a1\nideal -a1 phase=0 partner_flip=a2 partner_phase=90 deg\nideal a1"
    )
    events = program.build({"a1": np.pi / 2 * scp.ur.rad, "a2": "180 deg"})
    assert events == (
        pulsar.IdealPulse(90, 30),
        pulsar.IdealPulse(-90, 0, 180, 90),
        pulsar.IdealPulse(90, 30),
    )


@pytest.mark.parametrize(
    "text",
    [
        "ideal 1 us",
        "ideal 90 rf=10 kHz",
        "ideal 90 offset=1 Hz",
        "ideal 90 phase=1 Hz",
        "ideal 90 partner_flip=1 us",
        "ideal 90 partner_phase=1 kHz",
        "ideal 90 phase=0 phase=90",
        "ideal 1e309",
        "ideal missing",
    ],
)
def test_ideal_rejects_invalid_units_options_and_values_with_lines(text):
    with pytest.raises(pulsar.SequenceError) as caught:
        pulsar.parse_sequence("# header\n" + text, source="ideal.seq").build()
    assert caught.value.source == "ideal.seq"
    assert caught.value.line == 2


def test_simultaneous_ideal_rotations_match_coupled_density():
    events = pulsar.parse_sequence(
        "ideal a1 phase=ph1 partner_flip=a2 partner_phase=ph2"
    ).build({"a1": "60 deg", "ph1": 17, "a2": "120 deg", "ph2": 81})
    settings = {
        "observed": pulsar.SpinSite(0.5, 100e6),
        "partner": pulsar.SpinSite(0.5, 25e6),
        "j_hz": 120,
        "spectral_width_hz": 10000,
        "points": 128,
        "accuracy": 1,
        "output": "density",
    }
    actual = pulsar.simulate_coupled_sequence(events, **settings)
    reference = pulsar.simulate_coupled_sequence(
        [pulsar.IdealPulse(60, 17, 120, 81)], **settings
    )
    np.testing.assert_allclose(actual, reference, atol=1e-12)


@pytest.fixture(
    scope="module", params=["plot_ideal_nitrogen_mas", "plot_two_sodium_sites"]
)
def ideal_example(request):
    script = (
        Path(__file__).resolve().parents[1]
        / "examples/simulation"
        / f"{request.param}.py"
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(scp.NDDataset, "plot", lambda *a, **kw: None)
        patch.setattr(scp, "plot_multiple", lambda *a, **kw: None)
        phase = 37 if request.param == "plot_two_sodium_sites" else 0
        if phase:
            # Nonzero quadrature exposes an accidental real-only sum.
            load = pulsar.load_sequence
            patch.setattr(
                pulsar,
                "load_sequence",
                lambda name: SimpleNamespace(
                    build=lambda parameters: load(name).build(
                        {**parameters, "ph1": phase}
                    )
                ),
            )
        return request.param, runpy.run_path(str(script)), phase


def test_ideal_gallery_complex_parity_and_population_weights(ideal_example):
    name, example, phase = ideal_example
    if name == "plot_ideal_nitrogen_mas":
        reference = pulsar.simulate_pulse_sequence(
            [pulsar.IdealPulse(90)], **example["settings"]
        )
        np.testing.assert_allclose(example["spectrum"].data, reference.data, atol=1e-12)
        assert reference.sum().real == pytest.approx(1, abs=3e-5)
    else:
        references = [
            pulsar.simulate_pulse_sequence(
                [pulsar.IdealPulse(90, phase)], **example["settings"], **site
            )
            for site in example["sites"]
        ]
        for weight, actual, reference in zip(
            example["weights"], example["components"], references, strict=True
        ):
            np.testing.assert_allclose(actual.data, weight * reference.data, atol=1e-12)
        expected = 0.75 * references[0].data + references[1].data
        np.testing.assert_allclose(example["total"].data, expected, atol=1e-12)
        assert np.iscomplexobj(example["total"].data)
        assert np.max(np.abs(example["total"].data.imag)) > 1e-6
        assert example["weights"] == (0.75, 1.0)
