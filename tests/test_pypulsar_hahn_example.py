"""Physical checks kept outside the public Hahn echo gallery example."""

import runpy
from pathlib import Path

import numpy as np
import pytest
import spectrochempy as scp
from spectrochempy.nmr import pulsar


@pytest.fixture(scope="module")
def hahn_example():
    script = (
        Path(__file__).resolve().parents[1]
        / "examples/simulation/plot_pypulsar_hahn.py"
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(scp, "plot_multiple", lambda *args, **kwargs: None)
        return runpy.run_path(str(script))


@pytest.mark.parametrize("increment", range(4))
def test_hahn_example_complex_t2_attenuation(hahn_example, increment):
    example = hahn_example
    count = example["rotor_counts"][increment]
    events = example["hahn_events"](count)
    reference = pulsar.simulate_pulse_sequence(events, **example["settings"])
    elapsed = 2 * count * example["rotor_period"] + example["p90"] / 2
    expected = reference.data * np.exp(-elapsed / example["t2"])
    np.testing.assert_allclose(
        example["spectra"][increment].data, expected, rtol=1e-10, atol=1e-12
    )


def test_hahn_selected_pathway_recovers_complex_ideal_signal():
    settings = {
        "spin": 0.5,
        "larmor_hz": 100e6,
        "spectral_width_hz": 10e3,
        "shift_hz": 500,
        "accuracy": 2,
        "points": 1024,
    }
    reference = pulsar.simulate_pulse_sequence([pulsar.IdealPulse(90)], **settings)
    echo = pulsar.simulate_pulse_sequence(
        [
            pulsar.IdealPulse(90),
            pulsar.Coherence(-1),
            pulsar.Delay(100e-6),
            pulsar.IdealPulse(180, phase_deg=90),
            pulsar.Delay(100e-6),
        ],
        **settings,
    )
    np.testing.assert_allclose(echo.data, reference.data, atol=1e-12)
