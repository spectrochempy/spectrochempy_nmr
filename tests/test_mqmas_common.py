"""MQMAS operations consume physical States descriptors after reading."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import spectrochempy as scp

from qualification.conftest import paired_ser
from spectrochempy_nmr import from_interop
from spectrochempy_nmr.processing.fft_validation import _require_fft_prevalidation
from test_quadrature_contract import _acquisition


@pytest.fixture
def states_path(tmp_path):
    try:
        _require_fft_prevalidation()
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    path = paired_ser.__wrapped__(tmp_path, SimpleNamespace(param=4))
    for name in ("acqus", "acqu2s"):
        file = path / name
        file.write_text(file.read_text().replace("13C", "27Al"))
    return path


def _neutral(source):
    normalized, profile = _acquisition("states")
    first, second = source.data
    normalized.data = np.stack((first.real, first.imag, second.real, second.imag))
    normalized.axes[0].values = source.y.to("s").data.copy()
    normalized.axes[1].values = source.x.to("s").data.copy()
    normalized.acquisition = {
        key: tuple(source.meta.acquisition[key])
        for key in (
            "spectral_width_hz",
            "acquired_shape",
            "nuclei",
            "acquisition_frequency_mhz",
            "reference_frequency_mhz",
        )
    }
    normalized.acquisition["dc_offset"] = {"applied": False}
    normalized.acquisition["digital_filter"] = dict(
        source.meta.acquisition["digital_filter"]
    )
    normalized.normalization_profile = "neutral-physical-states"
    normalized.metadata = {}
    result = from_interop(normalized, profile=profile)
    result.units = source.units
    return result


def _workflow(source, *, points=16, window=None, corrected=True):
    options = {"echo_window": (19 / 12, 0.001)} if window == "gaussian" else {}
    if window == "mqecho":
        options["mqecho"] = (19 / 12, 40)
    mixed = scp.nmr.Experiment(source).xf2(
        size=256, direct_points=points, correct_digital_filter=corrected, **options
    )
    phased = scp.nmr.Experiment(mixed).xf2p(phc0=-31, phc1=44, pivot=2.5)
    spectrum = scp.nmr.Experiment(phased).xfshear(size=16, f1_shift=1.5)
    return mixed, phased, spectrum


@pytest.mark.parametrize("points", [2, 16, 128])
@pytest.mark.parametrize("window", [None, "gaussian", "mqecho"])
@pytest.mark.parametrize("corrected", [False, True])
def test_same_components_use_same_mqmas_operations(
    states_path, points, window, corrected
):
    source = scp.nmr.read(states_path)
    generic = _neutral(source)
    expected = _workflow(source, points=points, window=window, corrected=corrected)
    actual = _workflow(generic, points=points, window=window, corrected=corrected)
    for got, ref in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got.data, ref.data)
        np.testing.assert_array_equal(got.x.data, ref.x.data)
        np.testing.assert_array_equal(got.y.data, ref.y.data)


def test_reader_supplies_physical_counts_reference_and_delay(states_path):
    source = scp.nmr.read(states_path)
    acquisition = source.meta.acquisition
    assert acquisition["acquired_shape"] == (4, 128)
    assert acquisition["nuclei"] == ("27Al", "27Al")
    assert acquisition["acquisition_frequency_mhz"] == (125, 125)
    assert acquisition["reference_frequency_mhz"] == (124.999, 124.999)
    assert acquisition["digital_filter"]["group_delay_points"] == 2.5


def test_mqmas_ignores_identity_and_legacy_parameters_after_read(
    states_path, monkeypatch, tmp_path
):
    source = scp.nmr.read(states_path)
    expected = _workflow(source)
    source.write(tmp_path / "raw.scp", confirm=False)
    restored = scp.nmr.read(tmp_path / "raw.scp")
    restored.meta.readonly = False
    restored.origin = "other-vendor"
    restored.meta.normalization_profile = "unrelated-format"
    restored.meta.source_metadata = {"acqus": {"NUC1": "1H", "GRPDLY": 99}}
    restored.meta.pulprog = "unrelated-experiment"
    for key in ("sw_h", "sf", "sfo1", "tdeff", "nuc1"):
        restored.meta.data.pop(key, None)

    def unavailable(*args, **kwargs):
        pytest.fail("MQMAS processing must not reinterpret source files")

    provider = import_module("nmrglue.interop")
    monkeypatch.setattr(provider, "read_bruker_ser_storage", unavailable)
    monkeypatch.setattr(provider, "normalize_bruker_ser_states", unavailable)
    monkeypatch.setattr(
        import_module("spectrochempy_nmr.processing.bruker_filter"),
        "resolve_bruker_delay",
        unavailable,
    )
    before = restored.copy()
    actual = _workflow(restored)
    for got, ref in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got.data, ref.data)
        np.testing.assert_array_equal(got.x.data, ref.x.data)
        np.testing.assert_array_equal(got.y.data, ref.y.data)
    np.testing.assert_array_equal(restored.data, before.data)
    assert restored.meta == before.meta
    actual[1].meta.readonly = False
    actual[1].meta.source_metadata = {"acqus": None, "procs": "opaque provenance"}
    actual[1].write(tmp_path / "phased.scp", confirm=False)
    phased = scp.nmr.read(tmp_path / "phased.scp")
    final = scp.nmr.Experiment(phased).xfshear(size=16, f1_shift=1.5)
    np.testing.assert_array_equal(final.data, expected[-1].data)


@pytest.mark.parametrize("stored_delay", [None, -1, 0])
def test_no_decimation_is_resolved_at_read(states_path, stored_delay):
    acqus = states_path / "acqus"
    text = acqus.read_text().replace("##$DECIM= 16", "##$DECIM= 1")
    text = text.replace(
        "##$GRPDLY= 2.5\n",
        "" if stored_delay is None else f"##$GRPDLY= {stored_delay}\n",
    )
    acqus.write_text(text)
    source = scp.nmr.read(states_path)
    assert source.meta.acquisition["digital_filter"]["group_delay_points"] == 0
    source.meta.readonly = False
    source.meta.source_metadata = {}
    automatic = _workflow(source)[0]
    manual = _workflow(source, corrected=False)[0]
    np.testing.assert_array_equal(automatic.data, manual.data)


@pytest.mark.parametrize(
    "missing", ["reference_frequency_mhz", "acquired_shape", "spectral_width_hz"]
)
def test_missing_physical_context_refuses_before_transform(states_path, missing):
    source = scp.nmr.read(states_path)
    source.meta.readonly = False
    source.meta.acquisition.pop(missing)
    samples = source.data.copy()
    with pytest.raises(ValueError):
        _workflow(source)
    np.testing.assert_array_equal(source.data, samples)


def test_missing_delay_does_not_fall_back_to_source_parameters(states_path):
    source = scp.nmr.read(states_path)
    source.meta.readonly = False
    source.meta.acquisition["digital_filter"].pop("group_delay_points")
    before = source.data.copy()
    with pytest.raises(ValueError, match="normalized acquisition"):
        _workflow(source)
    np.testing.assert_array_equal(source.data, before)


def test_legacy_scp_metadata_is_completed_when_read(states_path, tmp_path):
    source = scp.nmr.read(states_path)
    expected = _workflow(source)[-1]
    source.meta.readonly = False
    for key in (
        "acquired_shape",
        "nuclei",
        "acquisition_frequency_mhz",
        "reference_frequency_mhz",
    ):
        source.meta.acquisition.pop(key)
    source.meta.acquisition["digital_filter"].pop("group_delay_points")
    source.write(tmp_path / "legacy.scp", confirm=False)
    restored = scp.nmr.read(tmp_path / "legacy.scp")
    result = _workflow(restored)[-1]
    np.testing.assert_array_equal(result.data, expected.data)


def test_unknown_quadrature_and_sampling_cannot_enable_mqmas(states_path):
    source = scp.nmr.read(states_path)
    source.meta.readonly = False
    source.meta.quadrature_mode = "undetermined"
    with pytest.raises(
        ValueError, match="normalized States|unsupported normalized interop structure"
    ):
        _workflow(source)
    source.meta.quadrature_mode = "states"
    source.y = scp.Coord([0, 0.001, 0.003, 0.004], units="s", rounding=False)
    with pytest.raises(ValueError, match="uniform acquisition grids"):
        _workflow(source)


def test_time_units_do_not_change_mqmas(states_path):
    source = scp.nmr.read(states_path)
    expected = _workflow(source)[-1]
    source.y.ito("us")
    source.x.ito("ms")
    actual = _workflow(source)[-1]
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(actual.x.data, expected.x.data)
    np.testing.assert_array_equal(actual.y.data, expected.y.data)


def test_legacy_phased_scp_can_resume_shearing(states_path, tmp_path):
    source = scp.nmr.read(states_path)
    _, phased, expected = _workflow(source)
    phased.meta.readonly = False
    for key in ("acquired_shape", "observed_nucleus"):
        phased.meta.mqmas_xf2.pop(key)
    for key in (
        "acquired_shape",
        "nuclei",
        "acquisition_frequency_mhz",
        "reference_frequency_mhz",
    ):
        phased.meta.acquisition.pop(key)
    phased.write(tmp_path / "old-phased.scp", confirm=False)
    restored = scp.nmr.read(tmp_path / "old-phased.scp")
    actual = scp.nmr.Experiment(restored).xfshear(size=16, f1_shift=1.5)
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(actual.y.data, expected.y.data)


def test_invalid_reference_fails_before_inplace_padding(states_path):
    source = scp.nmr.read(states_path)
    source.meta.readonly = False
    source.meta.acquisition["reference_frequency_mhz"] = (125, 0)
    samples = source.data.copy()
    with pytest.raises(ValueError, match="finite positive F2 reference"):
        source.fft(dim="x", size=256, convention="topspin-xf2", inplace=True)
    np.testing.assert_array_equal(source.data, samples)
    assert source.x.units == scp.ur.s


def test_sampling_schedule_cannot_be_treated_as_a_complete_grid(states_path):
    source = scp.nmr.read(states_path)
    source.meta.readonly = False
    source.meta.acquisition["sampling"] = ("listed", "regular")
    before = source.data.copy()
    with pytest.raises(ValueError, match="uniform acquisition sampling"):
        _workflow(source)
    np.testing.assert_array_equal(source.data, before)


@pytest.mark.parametrize("delay", [True, np.bool_(True), "2.5", -1, np.nan, np.inf])
def test_invalid_normalized_delay_refuses_before_processing(states_path, delay):
    source = scp.nmr.read(states_path)
    source.meta.readonly = False
    source.meta.acquisition["digital_filter"]["group_delay_points"] = delay
    before = source.data.copy()
    with pytest.raises(ValueError, match="finite nonnegative group delay"):
        _workflow(source)
    np.testing.assert_array_equal(source.data, before)


@pytest.mark.parametrize("legacy", [None, "contradictory"])
def test_final_spectrum_uses_normalized_physical_metadata(states_path, legacy):
    from spectrochempy_nmr.nmr_metadata import extract_nmr_metadata

    source = scp.nmr.read(states_path)
    source.meta.readonly = False
    if legacy is None:
        source.meta.data.pop("nuc1", None)
        source.meta.data.pop("sfo1", None)
        source.meta.data.pop("sw_h", None)
    else:
        source.meta.nuc1 = [None, "1H", "1H"]
        source.meta.sfo1 = [None, 99 * scp.ur.MHz, 99 * scp.ur.MHz]
        source.meta.sw_h = [None, 99 * scp.ur.Hz, 99 * scp.ur.Hz]
    for stage in (source, *_workflow(source)):
        physical = extract_nmr_metadata(stage.meta, ndim=stage.ndim)
        assert physical.nuclei == ("27Al", "27Al")
        assert physical.spectrometer_freq_mhz == (125, 125)
        assert physical.spectral_width_hz == (1000, 2000)
