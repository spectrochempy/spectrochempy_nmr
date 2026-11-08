"""Installed-provider NMRPipe bridge: synthetic CI and opt-in real references."""

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import from_interop
from spectrochempy_nmr.experiment import Experiment


@pytest.fixture(scope="module")
def provider():
    return pytest.importorskip("nmrglue", reason="optional external provider")


def check_experiment(
    provider, path, expected, hz, obs, complex_data, monkeypatch, tmp_path
):
    source_bytes = path.read_bytes()
    raw = provider.interop.read_nmrpipe_1d(path)
    normalized = provider.interop.normalize_nmrpipe_1d(raw)
    source = from_interop(normalized)
    before = deepcopy(dict(source.meta))

    def forbidden_fft(*args, **kwargs):
        raise AssertionError("frequency input must not FFT again")

    monkeypatch.setattr(scp.NDDataset, "fft", forbidden_fft, raising=False)
    result = Experiment(source).process()
    np.testing.assert_array_equal(result.data, expected)
    np.testing.assert_allclose(result.x.data, hz, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(
        result.x.to("ppm").data, hz / obs, rtol=1e-12, atol=1e-10
    )
    assert np.iscomplexobj(result.data) == complex_data
    assert Experiment(result).source_kind == "processed_1d"
    assert result.meta.nmr_processing["scp_processing"]["applied"] == {}
    changed = source.copy()
    changed.meta.readonly = False
    changed.origin = "unknown-vendor"
    changed.meta.normalization_profile = "opaque-profile"
    changed.meta.source_metadata = {}
    np.testing.assert_array_equal(Experiment(changed).process().data, expected)
    if complex_data:
        phased = Experiment(source).process(phase="manual", phc0=25)
        np.testing.assert_array_equal(phased.x.data, result.x.data)
        assert not np.array_equal(phased.data, expected)
    else:
        with pytest.raises(ValueError, match="imaginary"):
            Experiment(source).process(phase="manual", phc0=25)
    target = tmp_path / "nmrpipe.scp"
    result.write(target, confirm=False)
    restored = scp.read(target)
    np.testing.assert_array_equal(restored.data, expected)
    np.testing.assert_array_equal(restored.x.data, result.x.data)
    assert restored.meta.interop_axes[0]["reference"]["source"] == "processing"
    assert restored.meta.source_metadata["FDF2ORIG"] == raw.metadata["FDF2ORIG"]
    assert dict(source.meta) == before
    assert path.read_bytes() == source_bytes


@pytest.mark.parametrize("complex_data", [False, True])
def test_synthetic_nmrpipe_experiment(provider, complex_data, monkeypatch, tmp_path):
    header = provider.pipe.create_empty_dic()
    header.update(
        FDDIMCOUNT=1.0,
        FDDIMORDER1=2.0,
        FDSIZE=8.0,
        FDF2FTFLAG=1.0,
        FDF2SW=2000.0,
        FDF2OBS=400.0,
        FDF2ORIG=-625.0,
        FDF2CAR=99.0,
        FDF2QUADFLAG=float(not complex_data),
        FDQUADFLAG=float(not complex_data),
    )
    expected = np.arange(8, dtype=np.float32) - 2
    if complex_data:
        expected = expected + 1j * np.arange(8, dtype=np.float32)
    path = tmp_path / "synthetic.ft1"
    provider.pipe.write(str(path), header, expected)
    check_experiment(
        provider,
        path,
        expected,
        1125 - np.arange(8) * 250,
        400,
        complex_data,
        monkeypatch,
        tmp_path,
    )


@pytest.mark.parametrize("vendor", ["AGILENT", "BRUKER"])
@pytest.mark.parametrize("complex_data", [False, True])
def test_real_nmrpipe_external_ft(
    provider, vendor, complex_data, monkeypatch, tmp_path
):
    configured = os.environ.get("NMR_PIPE_" + vendor + "_SOURCE")
    executable = os.environ.get("NMR_PIPE_EXECUTABLE") or shutil.which("nmrPipe")
    if not configured or not executable:
        pytest.skip("opt-in real NMRPipe reference: configure source and executable")
    original = Path(configured)
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    expected_hash = {
        "AGILENT": "3cd39e31e3d4cfa0c04a01162a197b392a2315dc8a02fd5e3e6d2c4cfb57d380",
        "BRUKER": "b4ad9a4580f267f8763778b4b18b549bb25edf70374b70912e5dc9ca6a97a2a7",
    }[vendor]
    assert before == expected_hash, "configure the qualified historical reference FID"
    copied = tmp_path / "original.fid"
    shutil.copyfile(original, copied)
    spectrum = tmp_path / "complex.ft1"
    subprocess.run(
        [
            executable,
            "-in",
            str(copied),
            "-fn",
            "FT",
            "-auto",
            "-out",
            str(spectrum),
            "-ov",
        ],
        check=True,
        capture_output=True,
    )
    if not complex_data:
        real = tmp_path / "real.ft1"
        subprocess.run(
            [
                executable,
                "-in",
                str(spectrum),
                "-fn",
                "PS",
                "-p0",
                "0",
                "-p1",
                "0",
                "-di",
                "-out",
                str(real),
                "-ov",
            ],
            check=True,
            capture_output=True,
        )
        spectrum = real
    payload = spectrum.read_bytes()
    words = np.frombuffer(payload, dtype="<f4")
    assert words[2] == pytest.approx(2.345, rel=1e-6)
    size = int(words[99])
    assert size == {"AGILENT": 1500, "BRUKER": 2048}[vendor]
    assert float(words[100]) == {"AGILENT": 50000.0, "BRUKER": 10000.0}[vendor]
    assert float(words[119]) == pytest.approx(
        {"AGILENT": 125.68099975585938, "BRUKER": 600.1329956054688}[vendor]
    )
    assert words[220] == 1
    assert words[56] == float(not complex_data)
    assert len(words) == 512 + size * (2 if complex_data else 1)
    expected = words[512 : 512 + size].copy()
    if complex_data:
        expected = expected + 1j * words[512 + size :]
    hz = float(words[101]) + (size - 1 - np.arange(size)) * (float(words[100]) / size)
    obs = float(words[119])
    check_experiment(
        provider, spectrum, expected, hz, obs, complex_data, monkeypatch, tmp_path
    )
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before


def test_real_nmrpipe_cosy_plane(provider, tmp_path):
    configured = os.environ.get("NMR_PIPE_COSY_REFERENCE")
    if not configured:
        pytest.skip("configure the checksum-pinned COSY reference")
    path = Path(configured)
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "d13dc13c928394d2f8c7c57c0f65bdeab510d935093793406f53d7609ee70cb5"
    )
    words = np.frombuffer(payload, dtype="<f4")
    expected = words[512:].reshape(512, 2048)
    raw = provider.interop.read_nmrpipe_2d(path)
    dataset = from_interop(provider.interop.normalize_nmrpipe_2d(raw))
    assert dataset.meta.axis_roles == ("F1", "F2")
    assert dataset.dtype == expected.dtype
    np.testing.assert_array_equal(dataset.data, expected)
    # Header word offsets from NMRPipe fdata.h, independent of the provider axes.
    for coord, size, origin, width, obs in (
        (dataset.y, 512, words[249], words[229], words[218]),
        (dataset.x, 2048, words[101], words[100], words[119]),
    ):
        hz = float(origin) + (size - 1 - np.arange(size)) * (float(width) / size)
        assert coord.units == scp.ur.Hz
        np.testing.assert_allclose(coord.data, hz, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(
            coord.to("ppm").data, hz / obs, rtol=1e-12, atol=1e-10
        )
    target = tmp_path / "cosy.scp"
    dataset.write(target, confirm=False)
    restored = scp.read(target)
    np.testing.assert_array_equal(restored.data, expected)
    np.testing.assert_array_equal(restored.x.data, dataset.x.data)
    np.testing.assert_array_equal(restored.y.data, dataset.y.data)
    assert tuple(restored.meta.axis_roles) == ("F1", "F2")
    assert path.read_bytes() == payload


def _check_complex_nmrpipe(
    provider, path, expected_real, expected_imag, order, complex_dim, tmp_path
):
    raw = provider.interop.read_nmrpipe_2d_storage(path)
    normalized = provider.interop.normalize_nmrpipe_2d(raw)
    dataset = from_interop(normalized)
    assert dataset.meta.axis_roles == order
    assert dataset.meta.interop_complex_axis == complex_dim
    assert dataset.dtype == np.dtype("complex64")
    np.testing.assert_array_equal(dataset.data.real, expected_real)
    np.testing.assert_array_equal(dataset.data.imag, expected_imag)
    words = np.frombuffer(path.read_bytes(), dtype="<f4", count=512)
    offsets = {"F1": (249, 229, 218), "F2": (101, 100, 119)}
    for coord, dim, size in zip(
        (dataset.y, dataset.x), order, dataset.shape, strict=True
    ):
        origin, width, obs = (float(words[index]) for index in offsets[dim])
        hz = origin + (size - 1 - np.arange(size)) * (width / size)
        np.testing.assert_allclose(coord.data, hz, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(
            coord.to("ppm").data, hz / obs, rtol=1e-12, atol=1e-10
        )

    target = tmp_path / "complex-2d.scp"
    dataset.write(target, confirm=False)
    restored = scp.read(target)
    np.testing.assert_array_equal(restored.data, dataset.data)
    np.testing.assert_array_equal(restored.x.data, dataset.x.data)
    np.testing.assert_array_equal(restored.y.data, dataset.y.data)
    assert tuple(restored.meta.axis_roles) == order
    assert restored.meta.interop_complex_axis == complex_dim
    for original, reloaded in zip(
        (dataset.y, dataset.x), (restored.y, restored.x), strict=True
    ):
        assert reloaded.units == original.units
        assert (
            reloaded.meta.acquisition_frequency == original.meta.acquisition_frequency
        )


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("complex_dim", ["F1", "F2"])
def test_synthetic_nmrpipe_singly_complex_2d(
    provider, transposed, complex_dim, tmp_path
):
    header = provider.pipe.create_empty_dic()
    header.update(
        FDDIMCOUNT=2.0,
        FD2DPHASE=2.0,
        FDTRANSPOSED=float(transposed),
        FDDIMORDER1=1.0 if transposed else 2.0,
        FDDIMORDER2=2.0 if transposed else 1.0,
        FDSPECNUM=4.0 if transposed else 3.0,
        FDSIZE=3.0 if transposed else 4.0,
        FDQUADFLAG=0.0,
        FDF1QUADFLAG=float(complex_dim != "F1"),
        FDF2QUADFLAG=float(complex_dim != "F2"),
        FDF1FTFLAG=1.0,
        FDF2FTFLAG=1.0,
        FDF1FTSIZE=3.0,
        FDF2FTSIZE=4.0,
        FDF1SW=600.0,
        FDF2SW=2000.0,
        FDF1OBS=125.0,
        FDF2OBS=500.0,
        FDF1ORIG=-150.0,
        FDF2ORIG=-625.0,
    )
    real = np.arange(12, dtype="<f4").reshape(3, 4) - 5
    imag = 17 - 3 * real
    order = ("F2", "F1") if transposed else ("F1", "F2")
    if transposed:
        real, imag = real.T, imag.T
    if complex_dim == order[0]:
        payload = np.stack((real, imag), axis=1).reshape(2 * real.shape[0], -1)
    else:
        payload = np.concatenate((real, imag), axis=1)
    path = tmp_path / "one-complex-axis.ft2"
    path.write_bytes(
        provider.pipe.dic2fdata(header).astype("<f4").tobytes()
        + payload.astype("<f4").tobytes()
    )
    _check_complex_nmrpipe(provider, path, real, imag, order, complex_dim, tmp_path)


def test_nmrpipe_singly_complex_cosy_reference(provider, tmp_path):
    configured = os.environ.get("NMR_PIPE_COSY_REFERENCE")
    executable = os.environ.get("NMR_PIPE_EXECUTABLE") or shutil.which("nmrPipe")
    if not configured or not executable:
        pytest.skip("configure the checksum-pinned COSY input and nmrPipe executable")
    source = Path(configured)
    original = source.read_bytes()
    assert hashlib.sha256(original).hexdigest() == (
        "d13dc13c928394d2f8c7c57c0f65bdeab510d935093793406f53d7609ee70cb5"
    )
    path = tmp_path / "cosy-ht.ft2"
    subprocess.run(
        [
            executable,
            "-in",
            str(source),
            "-fn",
            "HT",
            "-auto",
            "-out",
            str(path),
            "-ov",
        ],
        check=True,
        capture_output=True,
    )
    words = np.frombuffer(path.read_bytes(), dtype="<f4")[512:].reshape(512, 4096)
    _check_complex_nmrpipe(
        provider, path, words[:, :2048], words[:, 2048:], ("F1", "F2"), "F2", tmp_path
    )
    assert source.read_bytes() == original
