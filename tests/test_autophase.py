import numpy as np
import pytest

from spectrochempy import Coord
from spectrochempy import NDDataset

from spectrochempy_nmr import autophase


@pytest.fixture
def spectrum():
    x = np.arange(2048, dtype=float)
    line = 1 / (1 + 1j * (x - 800) / 9)
    rng = np.random.default_rng(111)
    noise = 0.0002 * (rng.normal(size=x.size) + 1j * rng.normal(size=x.size))
    data = (line + noise + 0.025 + x * 1.5e-5) * np.exp(1j * np.deg2rad(37))
    result = NDDataset(data, coordset=[Coord(x, units="Hz")])
    result.meta.phased = [False]
    result.meta.exptc = [1]
    result.mask = (x >= 1200) & (x <= 1250)
    return result, line


def test_autophase_keeps_complex_intermediate_and_corrects_real_baseline(spectrum):
    source, line = spectrum
    data = source.data.copy()
    exclusion = np.zeros(data.shape, dtype=bool)
    exclusion[400:450] = True
    result = autophase(
        source,
        method="dispa",
        references=[(780, 821, 1)],
        exclude=exclusion,
        baseline_options={
            "model": "polynomial",
            "include_limits": False,
            "ranges": [(0, 300), (1700, 2047)],
        },
    )

    assert result.estimate.p0 == pytest.approx(-37, abs=0.8)
    assert result.phased.data == pytest.approx(
        data * np.exp(1j * np.deg2rad(result.estimate.p0))
    )
    np.testing.assert_array_equal(source.data, data)
    assert source.meta.phased == [False]
    followup = result.phased.pk(phc0=10, rel=True, pivot=0, exptc=0)
    np.testing.assert_allclose(
        followup.data, result.phased.data * np.exp(1j * np.deg2rad(10))
    )
    for output in (result.phased, result.baseline, result.corrected):
        np.testing.assert_array_equal(output.mask, source.mask)
        np.testing.assert_array_equal(output.x.data, source.x.data)
    assert result.corrected.is_complex is False
    valid = ~source.mask
    expected_baseline = 0.025 + source.x.data[valid] * 1.5e-5
    np.testing.assert_allclose(
        result.baseline.data[valid], expected_baseline, atol=0.002
    )
    region = slice(760, 841)
    assert np.sum(result.corrected.data[region]) == pytest.approx(
        np.sum(line.real[region]), rel=0.01
    )
    entry = result.phased.history_entries[-1]
    assert entry["operation"] == "nmr.autophase"
    assert entry["parameters"]["excluded_index_ranges"] == [[400, 450], [1200, 1251]]
    assert result.corrected.history_entries[-1]["operation"] == "baseline_correction"


def test_estimate_only_and_refuse_time_domain(spectrum):
    source, _ = spectrum
    result = autophase(source, apply=False)
    assert result.estimate.success
    assert result.phased is None
    assert source.history_entries == []
    source.meta.data_domain = "time"
    with pytest.raises(ValueError, match="frequency-domain"):
        autophase(source)


def test_automatic_asls_runs_on_the_phased_real_channel(spectrum):
    source, line = spectrum
    source.meta.acquisition = {"digital_filter": {"method": "pknl"}}
    source.meta.readonly = True
    result = autophase(
        source,
        method="dispa",
        references=[(780, 821, 1)],
        baseline_options={"model": "asls", "lamb": 1e9},
    )

    assert result.corrected.meta.readonly
    assert result.corrected.meta.acquisition["digital_filter"]["method"] == "pknl"
    assert result.corrected.history_entries[-1]["operation"] == "baseline_correction"
    anchors = (source.x.data < 300) | (source.x.data > 1700)
    residual = np.linalg.norm(result.corrected.data[anchors])
    uncorrected = np.linalg.norm(result.phased.data.real[anchors])
    assert residual < 0.1 * uncorrected
    region = slice(760, 841)
    assert np.sum(result.corrected.data[region]) == pytest.approx(
        np.sum(line.real[region]), rel=0.05
    )


def test_phase_metadata_references_this_call_for_already_phased_input(spectrum):
    source, _ = spectrum
    source.meta.phased = [True]
    source.meta.pivot = [50]
    result = autophase(source, method="dispa", references=[(780, 821, 1)])

    assert result.phased.meta.pivot == [0]
    assert result.phased.meta.exptc == [0]
    np.testing.assert_allclose(result.phased.pk(phc0=0).data, source.data)
    previous = result.phased.history_entries[-1]["parameters"][
        "previous_phase_metadata"
    ]
    assert previous["pivot"] == [50]
    assert previous["exptc"] == [1]


def test_autophase_uses_the_actual_1d_dimension(spectrum):
    source, _ = spectrum
    renamed = NDDataset(
        source.data, dims=["y"], coordset=[Coord(source.x.data, units="Hz")]
    )
    result = autophase(renamed, method="dispa", references=[(780, 821, 1)])
    assert result.estimate.p0 == pytest.approx(-37, abs=0.8)
    assert result.phased.dims == ["y"]
    np.testing.assert_array_equal(result.phased.y.data, renamed.y.data)

    spatial = NDDataset(
        source.data, dims=["y"], coordset=[Coord(source.x.data, units="m")]
    )
    with pytest.raises(ValueError, match="frequency-domain"):
        autophase(spatial)


def test_dispa_reference_generator_is_recorded(spectrum):
    source, _ = spectrum
    result = autophase(source, method="dispa", references=iter([(780, 821, 1)]))
    assert result.phased.history_entries[-1]["parameters"]["references"] == [
        [780, 821, 1]
    ]


@pytest.mark.parametrize("hidden", [np.nan, -1e5])
def test_asls_is_invariant_to_masked_sample_values(spectrum, hidden):
    source, _ = spectrum
    options = {
        "method": "dispa",
        "references": [(780, 821, 1)],
        "baseline_options": {"model": "asls", "lamb": 1e9},
    }
    expected = autophase(source, **options)
    changed = source.copy()
    data = source.data.copy()
    data[source.mask] = hidden
    changed.data = data
    result = autophase(changed, **options)

    assert result.estimate == expected.estimate
    valid = ~source.mask
    assert np.isfinite(result.corrected.data[valid]).all()
    for actual, reference in (
        (result.baseline, expected.baseline),
        (result.corrected, expected.corrected),
    ):
        np.testing.assert_array_equal(actual.mask, source.mask)
        np.testing.assert_allclose(actual.data[valid], reference.data[valid])
