"""Check the known signals behind the complete, synthetic gallery workflows."""

from pathlib import Path
import runpy

import matplotlib.pyplot as plt
import numpy as np
import pytest
import spectrochempy as scp
from spectrochempy.application.application import app


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def generated(tmp_path, monkeypatch):
    generator = runpy.run_path(str(ROOT / "docs" / "generate_reader_inputs.py"))
    root = generator["generate"](tmp_path)
    _ = scp.preferences.datadir
    monkeypatch.setitem(app.general_preferences._trait_values, "datadir", tmp_path)
    yield root
    plt.close("all")


@pytest.mark.parametrize(
    ("filename", "unit", "peak"),
    [
        ("plot_read_agilent", "ppm", 1.25),
        ("plot_read_simpson", "Hz", -500.0),
        ("plot_read_nmrml", "Hz", 500.0),
    ],
)
def test_reader_gallery_recovers_declared_sampling_and_peak(
    generated, filename, unit, peak
):
    if filename == "plot_read_nmrml":
        pytest.importorskip("xmltodict")
    path = ROOT / "examples" / "core" / "c_importer" / f"{filename}.py"
    result = runpy.run_path(str(path))
    raw, spectrum = result["raw"], result["spectrum"]
    times = np.arange(1024) / 4000.0
    expected = np.exp(-times / 0.04) * np.exp(2j * np.pi * 500 * times)
    np.testing.assert_allclose(raw.data, expected, atol=5e-8)
    np.testing.assert_allclose(raw.x.to("s").data, times, atol=1e-12)
    step = abs(np.diff(spectrum.x.data)[0])
    assert spectrum.x.units == scp.ur.Unit(unit)
    assert abs(spectrum.x.data[abs(spectrum).argmax()] - peak) <= step


def test_baseline_gallery_recovers_known_peak():
    result = runpy.run_path(str(ROOT / "examples/analysis/nmr/plot_nmr_baseline_2d.py"))
    np.testing.assert_allclose(result["corrected"].data, result["peak"], atol=1e-10)
    np.testing.assert_array_equal(
        result["spectrum"].data, result["peak"] + result["baseline"]
    )
    for axis in ("x", "y"):
        np.testing.assert_array_equal(
            result["corrected"].coord(axis).data,
            result["spectrum"].coord(axis).data,
        )
    plt.close("all")


def test_common_phase_gallery_recovers_signal_and_relative_amplitudes(generated):
    result = runpy.run_path(str(ROOT / "examples/analysis/nmr/plot_nmr_apk2d.py"))
    reference = result["reference"].data
    phased = result["phased"].data
    expected = np.array([1.0, 0.6, 0.3])[:, None] * reference
    np.testing.assert_allclose(phased, expected, atol=2e-3)
    np.testing.assert_allclose(
        result["series"].data,
        np.array([1.0, 0.6, 0.3])[:, None] * result["shifted"].data,
        rtol=2e-7,
    )
