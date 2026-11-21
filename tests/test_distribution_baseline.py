import importlib

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from spectrochempy import Baseline, Coord, NDDataset
from spectrochempy_nmr import autophase


HAS_DISTRIBUTION = "distribution" in Baseline.class_traits()["model"].values


@pytest.mark.skipif(
    not HAS_DISTRIBUTION, reason="core build lacks distribution baseline"
)
@pytest.mark.parametrize("width", [12, 31])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("descending", [False, True])
def test_autophase_distribution_preserves_complex_source_and_mask(
    masked, descending, width
):
    points = np.arange(4096.0)
    line = 1.0 / (1.0 + 1.0j * (points - 1800.0) / 6.0)
    background = 0.025 + points * 1.5e-5
    noise = np.random.default_rng(111).normal(0.0, 0.0002, points.size)
    values = (line + noise + background) * np.exp(1.0j * np.deg2rad(37.0))
    frequencies = points[::-1] if descending else points
    source = NDDataset(
        values, coordset=[Coord(frequencies, units="Hz")], units="absorbance"
    )
    source.meta.phased = [False]
    source.meta.exptc = [1]
    if masked:
        source.mask = (points >= 2500) & (points <= 2520)
        source.data[source.mask] = complex(np.nan, np.nan)
    original = source.data.copy()
    result = autophase(
        source,
        method="dispa",
        references=[(1780, 1821, 1)],
        baseline_options={"model": "distribution", "distribution_width": width},
    )
    assert result.estimate.success
    assert result.phased.is_complex
    assert not result.corrected.is_complex
    assert_allclose(
        result.phased.data,
        original * np.exp(1.0j * np.deg2rad(result.estimate.p0)),
        equal_nan=True,
    )
    assert_array_equal(source.data, original)
    assert source.meta.phased == [False]
    assert source.history_entries == []
    for output in (result.phased, result.baseline, result.corrected):
        assert output.shape == source.shape
        assert output.units == source.units
        assert_array_equal(output.x.data, source.x.data)
        assert_array_equal(output.mask, source.mask)
    valid = ~np.broadcast_to(source.mask, source.shape)
    assert np.isfinite(result.corrected.data[valid]).all()
    anchors = valid & (abs(points - 1800) > 200)
    assert (
        np.sqrt(np.mean((result.baseline.data[anchors] - background[anchors]) ** 2))
        < 0.002
    )
    region = abs(points - 1800) <= 30
    assert np.sum(result.corrected.data[region]) == pytest.approx(
        np.sum(line.real[region]), rel=0.03
    )
    entry = result.corrected.history_entries[-1]
    assert entry["operation"] == "baseline_correction"
    assert entry["parameters"]["experimental"] is True
    assert entry["parameters"]["distribution_width"] == width
    assert entry["parameters"]["masked_sample_fill"] == (
        "linear_index" if masked else None
    )


def test_distribution_requires_a_capable_core_before_phase_estimation(monkeypatch):
    module = importlib.import_module("spectrochempy_nmr.processing.autophase")

    class OldBaseline:
        @classmethod
        def class_traits(cls):
            class Model:
                values = ["polynomial"]

            return {"model": Model()}

    monkeypatch.setattr(module, "Baseline", OldBaseline)
    source = NDDataset(
        np.ones(16, dtype=complex), coordset=[Coord(np.arange(16.0), units="Hz")]
    )
    with pytest.raises(ValueError, match="core|SpectroChemPy build"):
        autophase(source, baseline_options={"model": "distribution"})
    assert source.history_entries == []
