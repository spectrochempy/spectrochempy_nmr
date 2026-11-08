"""FFT and reconstruction of imported echo/anti-echo against NMRPipe."""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import from_interop
from spectrochempy_nmr.experiment import Experiment
from spectrochempy_nmr.processing import channel_interop
from spectrochempy_nmr.readers.read_topspin import _remove_digital_filter
from test_interop_conversion import _normalized

PROFILE = "bruker-rance-kay"
ORACLE = Path(__file__).parent / "data" / "nmrpipe" / "echo_antiecho_oracle.npz"


def _assert_fft(actual, expected, tolerance):
    # NMRPipe checkpoints are single precision. Check each plane independently.
    for plane, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            plane, reference, rtol=0, atol=tolerance * np.max(np.abs(reference))
        )


def _qualify(source, oracle, monkeypatch, *, group_delay=None):
    observed = {}
    reconstruct = channel_interop._rance_kay
    pair = channel_interop._cross_pairs

    def record_reconstruction(data):
        shuffled = reconstruct(data)
        observed["shuffled"] = shuffled.copy()
        return shuffled

    def record_cross(data):
        observed["f2"] = data.copy()
        cross = pair(data)
        observed["cross"] = cross.copy()
        return cross

    monkeypatch.setattr(channel_interop, "_rance_kay", record_reconstruction)
    monkeypatch.setattr(channel_interop, "_cross_pairs", record_cross)
    experiment = Experiment(source)
    assert experiment.source_kind == "ser"
    result = experiment.process(echo_antiecho=PROFILE, bruker_group_delay=group_delay)
    # NMRPipe uses single precision for both the FFT and the FT -df phase ramp.
    tolerance = 2e-5 if group_delay is not None else 1e-6

    np.testing.assert_array_equal(source.data, oracle["raw"])
    np.testing.assert_array_equal(observed["shuffled"], oracle["shuffled"])
    for checkpoint in ("f2", "cross"):
        _assert_fft(observed[checkpoint], oracle[checkpoint], tolerance)
    _assert_fft(result.data, oracle["spectrum"], tolerance)
    assert tuple(result.c.labels) == ("rr", "ri", "ir", "ii")
    assert result.y.units == result.x.units == scp.ur.Hz
    for dim, coord in (("f1", result.y), ("f2", result.x)):
        np.testing.assert_allclose(coord.data, oracle[f"hz_{dim}"], atol=5e-4, rtol=0)
    assert Experiment(result).source_kind == "processed_2d"
    assert source.meta.echo_antiecho_reconstruction is None
    assert result.meta.echo_antiecho_reconstruction == PROFILE
    trace = result.meta.nmr_processing["scp_processing"]
    assert trace["requested"]["echo_antiecho"] == PROFILE
    assert trace["applied"]["echo_antiecho"] == PROFILE
    if group_delay is not None:
        assert trace["requested"]["bruker_group_delay"] == group_delay
        assert trace["applied"]["bruker_group_delay"] == group_delay
        assert trace["applied"]["fft_convention"][0] == "NMRPipe FT -df"
        assert result.shape[-1] == source.shape[-1]
    return result


@pytest.mark.parametrize("digital_filter", [False, True])
def test_synthetic_echo_antiecho_checkpoints(monkeypatch, tmp_path, digital_filter):
    path = (
        ORACLE.with_name("echo_antiecho_dmx_oracle.npz") if digital_filter else ORACLE
    )
    checksum = (
        "57fa1830dfb13482b852ed053af97e64a990afc3e45f6e270b4b4113404595ff"
        if digital_filter
        else "f7c48b0b4b3489c85445441090faaaab7ee2ebd7a2b481d8be555185c52dd99d"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum
    with np.load(path) as oracle:
        data = oracle["raw"]
        f1 = _normalized(axis_values=np.arange(data.shape[1]) / 1000).axes[0]
        normalized = _normalized(data, axis_values=np.arange(data.shape[2]) / 4000)
        normalized.axes = (f1, *normalized.axes)
        normalized.dimensions = ("channels", "F1", "F2")
        normalized.physical_dimensions = ("F1", "F2")
        normalized.domains = ("time", "time")
        normalized.quadrature_mode = "echo-antiecho-stored-2d"
        normalized.branch_order = ("first", "second")
        normalized.channel_axis = 0
        normalized.encoding = ("echo-antiecho", "complex")
        normalized.acquisition = {"spectral_width_hz": (1000, 4000)}
        source = from_interop(normalized)

        experiment = Experiment(source)
        with pytest.raises(ValueError, match="requires echo_antiecho"):
            experiment.process()
        with pytest.raises(ValueError, match="supports FFT only"):
            experiment.process(echo_antiecho=PROFILE, size=256)
        for delay in (0, -1, np.inf, np.nan, True, "3", data.shape[-1]):
            with pytest.raises(ValueError, match="group delay"):
                experiment.process(echo_antiecho=PROFILE, bruker_group_delay=delay)
        with pytest.raises(ValueError, match="Unknown Bruker group delay"):
            experiment.process(echo_antiecho=PROFILE, digital_filter="phase")
        with pytest.raises(ValueError, match="fewer than two"):
            experiment.process(
                echo_antiecho=PROFILE, digital_filter="time", bruker_group_delay=126
            )

        if digital_filter:
            source.meta.source_metadata = {
                "acqus": {"GRPDLY": 2.5, "DSPFVS": 12, "DECIM": 24}
            }
            automatic = Experiment(source).process(
                echo_antiecho=PROFILE, digital_filter="phase"
            )
            manual = Experiment(source).process(
                echo_antiecho=PROFILE, bruker_group_delay=2.5
            )
            np.testing.assert_array_equal(automatic.data, manual.data)
            assert automatic.meta.bruker_digital_filter["delay_source"] == "GRPDLY"
        corrected = source.copy()
        corrected.meta.acquisition = {
            **normalized.acquisition,
            "digital_filter": {"applied": True},
        }
        with pytest.raises(ValueError, match="already been corrected"):
            Experiment(corrected).process(echo_antiecho=PROFILE, bruker_group_delay=2)
        result = _qualify(
            source,
            oracle,
            monkeypatch,
            group_delay=70 + 1 / 6 if digital_filter else None,
        )

    path = tmp_path / "echo-spectrum.scp"
    result.write(path, confirm=False)
    restored = scp.read(path)
    assert Experiment(restored).source_kind == "processed_2d"
    np.testing.assert_array_equal(restored.data, result.data)
    with pytest.raises(ValueError, match="requires an imported raw"):
        Experiment(restored).process(echo_antiecho=PROFILE)
    with pytest.raises(ValueError, match="bruker_group_delay requires"):
        Experiment(restored).process(bruker_group_delay=2)
    if digital_filter:
        assert (
            restored.meta.nmr_processing["scp_processing"]["applied"][
                "bruker_group_delay"
            ]
            == 70 + 1 / 6
        )


@pytest.mark.parametrize("digital_filter", [False, True])
def test_official_hsqc_echo_antiecho_checkpoints(monkeypatch, digital_filter):
    xml = os.environ.get("NMRML_HSQC_XML")
    oracle_variable = "NMRML_HSQC_DMX_ORACLE" if digital_filter else "NMRML_HSQC_ORACLE"
    path = os.environ.get(oracle_variable)
    pdata = os.environ.get("NMRML_HSQC_PDATA")
    if not xml or not path or not pdata:
        pytest.skip(f"set NMRML_HSQC_XML, {oracle_variable} and NMRML_HSQC_PDATA")
    pytest.importorskip("xmltodict")
    ng = pytest.importorskip("nmrglue")
    assert hashlib.sha256(Path(xml).read_bytes()).hexdigest() == (
        "3231a4769029f362863fb57c7adaaaf6aae304a65c503839b57e5ec02b320f82"
    )
    raw = ng.interop.read_nmrml_raw_2d(
        xml, direct_point_count="scalar", official_hsqc_compatibility=True
    )
    normalized = ng.interop.normalize_nmrml_raw_2d(
        raw, indirect_dwell_s=132.6e-6, confirm_adjacent_pairs=True
    )
    source = from_interop(normalized)
    with np.load(path) as oracle:
        delay = 70 + 1 / 6 if digital_filter else None
        if digital_filter:
            assert float(oracle["group_delay"]) == pytest.approx(delay, abs=3e-6)
        result = _qualify(source, oracle, monkeypatch, group_delay=delay)

    # The two resolved direct bands distinguish the correct F1 placement from
    # a half-width shift caused by applying an additional FT -alt.
    directory = Path(pdata)
    assert hashlib.sha256((directory / "2rr").read_bytes()).hexdigest() == (
        "7108562f23487c82097f5841f3a35af64394fbcbf99bd2ca6bd8d24a7d7ed5ae"
    )
    _, native = ng.bruker.read_pdata(directory, scale_data=False)
    magnitude = np.linalg.norm(result.data, axis=0)
    assert native.shape == (2 * magnitude.shape[0], 2 * magnitude.shape[1])
    for column in (1299, 1537):
        processed = magnitude[:, column - 5 : column + 6]
        native_band = native[:, 2 * column - 10 : 2 * column + 12]
        f1 = np.unravel_index(processed.argmax(), processed.shape)[0]
        f1_native = np.unravel_index(np.abs(native_band).argmax(), native_band.shape)[0]
        assert abs(2 * f1 - f1_native) <= 2

    if digital_filter:
        acqus_path = os.environ.get("NMRML_HSQC_ACQUS")
        if not acqus_path:
            pytest.skip("set NMRML_HSQC_ACQUS to qualify automatic Bruker correction")
        acqus_file = Path(acqus_path)
        assert hashlib.sha256(acqus_file.read_bytes()).hexdigest() == (
            "69fe04b277990aa0076042a012a7549473a5547c12c52c53c0ea30fb22210e1a"
        )
        metadata = {"acqus": ng.bruker.read_jcamp(acqus_file)}
        source.meta.source_metadata = metadata
        # nmrglue-ng's positive post-FFT phase ramp requires the conjugation
        # adapter documented in nmrglue-ng #194 for the positive-exponent FFT.
        with np.load(path) as oracle:
            f2 = ng.proc_base.fft_positive(oracle["shuffled"].astype(np.complex128))
        expected_f2 = ng.bruker.remove_digital_filter(
            metadata, f2.conj(), truncate=False, post_proc=True
        ).conj()
        cross_pairs = channel_interop._cross_pairs

        def check_provider_f2(f2):
            _assert_fft(f2, expected_f2, 1e-12)
            return cross_pairs(f2)

        with monkeypatch.context() as check:
            check.setattr(channel_interop, "_cross_pairs", check_provider_f2)
            automatic = Experiment(source).process(
                echo_antiecho=PROFILE, digital_filter="phase"
            )
        np.testing.assert_array_equal(automatic.data, result.data)
        assert automatic.meta.bruker_digital_filter["delay_source"] == "DSPFVS/DECIM"

        expected = np.stack(
            [_remove_digital_filter(metadata, branch) for branch in source.data]
        )
        reconstruct = channel_interop._rance_kay

        def check_branches(data):
            np.testing.assert_array_equal(data, expected)
            return reconstruct(data)

        monkeypatch.setattr(channel_interop, "_rance_kay", check_branches)
        time = Experiment(source).process(echo_antiecho=PROFILE, digital_filter="time")
        assert time.shape == (4, 128, 1976)
        assert time.meta.bruker_digital_filter["removed_points"] == 72
        assert time.meta.bruker_digital_filter["delay_source"] == "DSPFVS/DECIM"
        np.testing.assert_allclose(
            time.x.data, (988 - np.arange(1976)) * (7002.80112044818 / 1976)
        )
