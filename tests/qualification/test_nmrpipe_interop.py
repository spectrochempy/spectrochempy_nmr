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
