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
from test_interop_conversion import _normalized

PROFILE = "bruker-rance-kay"
ORACLE = Path(__file__).parent / "data" / "nmrpipe" / "echo_antiecho_oracle.npz"


def _assert_fft(actual, expected):
    # NMRPipe checkpoints are single precision. Check each plane independently.
    for plane, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            plane, reference, rtol=0, atol=1e-6 * np.max(np.abs(reference))
        )


def _qualify(source, oracle, monkeypatch):
    observed = {}
    fft = channel_interop._fft_stage
    reconstruct = channel_interop._rance_kay
    pair = channel_interop._cross_pairs

    def record_reconstruction(data):
        shuffled = reconstruct(data)
        observed["shuffled"] = shuffled.copy()
        return shuffled

    def record_fft(data, axis, *, direct):
        output = fft(data, axis, direct=direct)
        if axis == -1:
            observed["f2"] = output.copy()
        return output

    def record_cross(data):
        cross = pair(data)
        observed["cross"] = cross.copy()
        return cross

    monkeypatch.setattr(channel_interop, "_rance_kay", record_reconstruction)
    monkeypatch.setattr(channel_interop, "_fft_stage", record_fft)
    monkeypatch.setattr(channel_interop, "_cross_pairs", record_cross)
    experiment = Experiment(source)
    assert experiment.source_kind == "ser"
    result = experiment.process(echo_antiecho=PROFILE)

    np.testing.assert_array_equal(source.data, oracle["raw"])
    np.testing.assert_array_equal(observed["shuffled"], oracle["shuffled"])
    for checkpoint in ("f2", "cross"):
        _assert_fft(observed[checkpoint], oracle[checkpoint])
    _assert_fft(result.data, oracle["spectrum"])
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
    return result


def test_synthetic_echo_antiecho_checkpoints(monkeypatch, tmp_path):
    assert hashlib.sha256(ORACLE.read_bytes()).hexdigest() == (
        "f7c48b0b4b3489c85445441090faaaab7ee2ebd7a2b481d8be555185c52dd99d"
    )
    with np.load(ORACLE) as oracle:
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
        result = _qualify(source, oracle, monkeypatch)

    path = tmp_path / "echo-spectrum.scp"
    result.write(path, confirm=False)
    restored = scp.read(path)
    assert Experiment(restored).source_kind == "processed_2d"
    np.testing.assert_array_equal(restored.data, result.data)
    with pytest.raises(ValueError, match="requires an imported raw"):
        Experiment(restored).process(echo_antiecho=PROFILE)


def test_official_hsqc_echo_antiecho_checkpoints(monkeypatch):
    xml = os.environ.get("NMRML_HSQC_XML")
    path = os.environ.get("NMRML_HSQC_ORACLE")
    pdata = os.environ.get("NMRML_HSQC_PDATA")
    if not xml or not path or not pdata:
        pytest.skip("set NMRML_HSQC_XML, NMRML_HSQC_ORACLE and NMRML_HSQC_PDATA")
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
    with np.load(path) as oracle:
        result = _qualify(from_interop(normalized), oracle, monkeypatch)

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
