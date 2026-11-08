"""Installed nmrML provider through generic interop/Experiment, without routing."""

from copy import deepcopy
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import from_interop
from spectrochempy_nmr.experiment import Experiment

FIXTURE = Path(__file__).parent / "fixtures" / "nmrml_synthetic.nmrML"


@pytest.fixture(scope="module")
def provider():
    pytest.importorskip("xmltodict")
    return pytest.importorskip("nmrglue")


def check_frequency(provider, raw, expected, tmp_path, monkeypatch):
    normalized = provider.interop.normalize_nmrml_1d(raw)
    source = from_interop(normalized)
    before = deepcopy(dict(source.meta))

    def no_fft(*args, **kwargs):
        raise AssertionError("processed nmrML must not FFT again")

    monkeypatch.setattr(scp.NDDataset, "fft", no_fft, raising=False)
    result = Experiment(source).process()
    np.testing.assert_array_equal(result.data, expected)
    np.testing.assert_allclose(
        result.x.data, raw.axes[0].values, rtol=1e-12, atol=1e-12
    )
    if raw.axes[0].reference.observation_frequency is not None:
        obs = raw.axes[0].reference.observation_frequency
        np.testing.assert_allclose(result.x.to("ppm").data, raw.axes[0].values / obs)
    changed = source.copy()
    changed.meta.readonly = False
    changed.origin = "opaque-origin"
    changed.meta.normalization_profile = "opaque-profile"
    changed.meta.source_metadata = {}
    np.testing.assert_array_equal(Experiment(changed).process().data, expected)
    path = tmp_path / "normalized.scp"
    result.write(path, confirm=False)
    restored = scp.read(path)
    np.testing.assert_array_equal(restored.data, expected)
    np.testing.assert_array_equal(restored.x.data, result.x.data)
    assert (
        restored.meta.source_metadata["profile_options"]["axis_endpoint"]
        == raw.acquisition["axis_endpoint"]
    )
    assert restored.meta.interop_axes[0]["reference"]["source"] == "unknown"
    assert dict(source.meta) == before


@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("obs", [None, 400.0])
def test_synthetic_processed_nmrml(provider, tmp_path, monkeypatch, complex_data, obs):
    document, _ = provider.nmrml.read(FIXTURE, read_processed=True, strict=True)
    values = np.arange(8, dtype="f8")
    if complex_data:
        values = values + 1j * np.arange(8, dtype="f8")
    path = tmp_path / "spectrum.nmrML"
    provider.nmrml.write(path, document, values, write_processed=True)
    raw = provider.interop.read_nmrml_1d(
        path,
        read_processed=True,
        axis_endpoint="last-point",
        observation_frequency_mhz=obs,
    )
    check_frequency(provider, raw, values, tmp_path, monkeypatch)


@pytest.mark.parametrize("sign", ["none", "conjugate"])
def test_synthetic_raw_nmrml(provider, tmp_path, sign):
    document, _ = provider.nmrml.read(FIXTURE, strict=True)
    values = np.exp(2j * np.pi * 0.125 * np.arange(8)).astype("complex128")
    path = tmp_path / "fid.nmrML"
    provider.nmrml.write(path, document, values)
    raw = provider.interop.read_nmrml_1d(path, raw_point_count="scalar")
    normalized = provider.interop.normalize_nmrml_1d(raw, conjugation=sign)
    source = from_interop(normalized)
    expected = values.conj() if sign == "conjugate" else values
    np.testing.assert_array_equal(source.data, expected)
    np.testing.assert_array_equal(source.x.data, np.arange(8) / 2000)
    result = Experiment(source).process()
    assert Experiment(result).source_kind == "processed_1d"
    assert result.shape == expected.shape
    analytical = np.zeros(8, dtype="complex128")
    peak = 5 if sign == "none" else 3
    analytical[peak] = 8
    np.testing.assert_allclose(result.data, analytical, rtol=0, atol=1e-14)
    assert result.x.data[peak] == (250 if sign == "none" else -250)
    np.testing.assert_array_equal(source.data, expected)
    persisted = tmp_path / "fid.scp"
    source.write(persisted, confirm=False)
    restored = scp.read(persisted)
    np.testing.assert_array_equal(restored.data, expected)
    np.testing.assert_array_equal(restored.x.data, source.x.data)


def test_real_bayesil_nmrml(provider, tmp_path, monkeypatch):
    root = os.environ.get("NMRML_QUALIFICATION_DATA")
    if not root:
        pytest.skip(
            "set NMRML_QUALIFICATION_DATA to the checksum-pinned public examples"
        )
    path = Path(root) / "biosample_quantification.nmrML"
    original = path.read_bytes()
    assert (
        hashlib.sha256(original).hexdigest()
        == "38712b01526635b3bf2639a5b08c7e00e15c4e56d0c1174295595b9ab939c448"
    )
    raw = provider.interop.read_nmrml_1d(
        path, read_processed=True, axis_endpoint="last-point"
    )
    expected = provider.nmrml.read(path, read_processed=True, strict=True)[1]
    check_frequency(provider, raw, expected, tmp_path, monkeypatch)
    assert path.read_bytes() == original


def test_real_hop_nmrml(provider, tmp_path):
    root = os.environ.get("NMRML_QUALIFICATION_DATA")
    if not root:
        pytest.skip(
            "set NMRML_QUALIFICATION_DATA to official nmrML and native examples"
        )
    path = Path(root) / "hop.nmrML"
    original = path.read_bytes()
    assert (
        hashlib.sha256(original).hexdigest()
        == "2f8a0b7cdc4ca8083aecb2df1ab6d81fbede5891ee2c3b562cc8dd079f8837d0"
    )
    raw = provider.interop.read_nmrml_1d(
        path, data_dtype="<c8", raw_point_count="scalar"
    )
    _, expected = provider.varian.read(Path(root) / "hop-native")
    source = from_interop(provider.interop.normalize_nmrml_1d(raw, conjugation="none"))
    np.testing.assert_array_equal(source.data, expected)
    np.testing.assert_allclose(source.x.data, np.arange(expected.size) / 12019.2307692)
    result = Experiment(source).process()
    assert Experiment(result).source_kind == "processed_1d"
    assert path.read_bytes() == original


def test_official_hsqc_echo_antiecho(provider, tmp_path):
    xml = os.environ.get("NMRML_HSQC_XML")
    ser = os.environ.get("NMRML_HSQC_SER")
    if not xml or not ser:
        pytest.skip("set NMRML_HSQC_XML and NMRML_HSQC_SER to the reference files")
    assert hashlib.sha256(Path(xml).read_bytes()).hexdigest() == (
        "3231a4769029f362863fb57c7adaaaf6aae304a65c503839b57e5ec02b320f82"
    )
    raw = Path(ser).read_bytes()
    assert hashlib.sha1(raw).hexdigest() == "a979ed42e5a28066a476557d713c76ce081541da"
    words = np.frombuffer(raw, dtype=">i4").reshape(256, 4096)
    traces = words[:, ::2] + 1j * words[:, 1::2]

    stored = provider.interop.read_nmrml_raw_2d(
        xml, direct_point_count="scalar", official_hsqc_compatibility=True
    )
    dwell = 2 * 6.63e-05
    normalized = provider.interop.normalize_nmrml_raw_2d(
        stored, indirect_dwell_s=dwell, confirm_adjacent_pairs=True
    )
    source = from_interop(normalized)
    path = tmp_path / "hsqc.scp"
    source.write(path, confirm=False)
    restored = scp.read(path)
    for dataset in (source, restored):
        assert dataset.dims == ["c", "y", "x"]
        assert dataset.shape == (2, 128, 2048)
        assert dataset.dtype == np.complex128
        assert tuple(dataset.meta.branch_order) == ("first", "second")
        assert dataset.meta.quadrature_mode == "echo-antiecho-stored-2d"
        np.testing.assert_array_equal(dataset.c.labels, ["first", "second"])
        np.testing.assert_array_equal(dataset.data[0], traces[::2])
        np.testing.assert_array_equal(dataset.data[1], traces[1::2])
        assert dataset.y.units == dataset.x.units == scp.ur.s
        np.testing.assert_allclose(dataset.y.data, np.arange(128) * dwell, rtol=1e-12)
        np.testing.assert_allclose(
            dataset.x.data, np.arange(2048) / 7002.80112044818, rtol=1e-12
        )
