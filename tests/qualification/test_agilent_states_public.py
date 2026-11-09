"""Qualify the public Agilent States reader against its compatibility path."""

import hashlib
import os
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
from nmrglue import interop
from nmrglue import varian

import spectrochempy as scp
import spectrochempy_nmr.interop as adapter


def _check_states(path, monkeypatch):
    convert = adapter.from_interop
    calls = []

    def spy(result):
        dataset = convert(result)
        calls.append(dataset)
        return dataset

    def reject_fallback(*args, **kwargs):
        pytest.fail("qualified States called the historical decoder")

    reader = import_module("spectrochempy_nmr.readers.read_agilent")
    with monkeypatch.context() as baseline:
        baseline.setattr(interop, "normalize_agilent_states_2d", None)
        old = scp.read_agilent(path, origin="qualification")

    monkeypatch.setattr(adapter, "from_interop", spy)
    monkeypatch.setattr(reader, "read_varian", reject_fallback)
    actual = scp.read_agilent(path, origin="qualification")
    assert len(calls) == 1 and actual is calls[0]
    assert actual.dims == old.dims
    assert actual.dtype == old.dtype
    assert actual.origin == old.origin == "qualification"
    assert actual.units == old.units
    assert actual.meta.normalization_state == "storage-normalized"
    for key, expected in old.meta.items():
        if key != "normalization_state":
            np.testing.assert_equal(actual.meta[key], expected, err_msg=key)
    np.testing.assert_array_equal(actual.data, old.data)
    for dim in actual.dims:
        got, expected = actual.coordset[dim], old.coordset[dim]
        assert got.title == expected.title
        assert got.units == expected.units
        assert got.meta == expected.meta
        np.testing.assert_allclose(got.data, expected.data, rtol=1e-12, atol=0)
        np.testing.assert_array_equal(got.labels, expected.labels)

    direct, baseline = actual.fft(), old.fft()
    np.testing.assert_array_equal(direct.data, baseline.data)
    np.testing.assert_array_equal(direct.x.data, baseline.x.data)
    assert direct.x.units == baseline.x.units
    with pytest.raises(NotImplementedError, match="UNVALIDATED:AGILENT-STATES"):
        direct.fft(dim="y")
    return actual


def test_synthetic_states_public(tmp_path, monkeypatch):
    params = {
        "array": ["phase"],
        "phase": ["1", "2"],
        "ni": ["2"],
        "arraydim": ["4"],
        "np": ["8"],
        "sw": ["2000"],
        "sw1": ["500"],
        "sw2": ["800"],
        "sfrq": ["125"],
        "dfrq2": ["50"],
        "ns": ["3"],
        "ne": ["2"],
    }
    varian.write_procpar(
        tmp_path / "procpar",
        {key: varian.create_pdic_param(key, values) for key, values in params.items()},
    )
    words = np.arange(32, dtype=">i4").reshape(4, 8)
    words[:, 1::2] *= -7
    with (tmp_path / "fid").open("wb") as stream:
        varian.put_fileheader(stream, (4, 1, 8, 4, 32, 60, 0, 21, 1))
        for index, trace in enumerate(words, 1):
            varian.put_blockheader(stream, (0, 21, index, 0, 3, 0, 0, 0, 0))
            stream.write(trace.tobytes())
    actual = _check_states(tmp_path / "fid", monkeypatch)
    samples = words[:, ::2] + 1j * words[:, 1::2]
    np.testing.assert_array_equal(actual.data[0], samples[::2] / 6)
    np.testing.assert_array_equal(actual.data[1], samples[1::2] / 6)
    scales = [
        operation
        for operation in actual.meta.applied_operations
        if operation["name"] == "normalize-agilent-ns-ne"
    ]
    assert len(scales) == 1 and scales[0]["divisor"] == 6


def test_real_states_public(monkeypatch):
    directory = os.environ.get("NMR_AGILENT_STATES_DATA")
    if not directory:
        pytest.skip(
            "set NMR_AGILENT_STATES_DATA to the historical agilent_2d directory"
        )
    path = Path(directory)
    files = (path / "fid", path / "procpar")
    before = tuple(hashlib.sha256(file.read_bytes()).hexdigest() for file in files)
    assert before == (
        "4a35f4b98f9116ba67760d1c0c83cc48a17b4e53897b5dbe1e25f85f28b1eac8",
        "54cf12841987ec3a05f5ca76f2b09dff09b7648d84b2936627f2c9757472c817",
    )
    actual = _check_states(path, monkeypatch)
    assert actual.shape == (2, 166, 1500)
    assert actual.y.units is None
    assert actual.meta.acquisition["spectral_widths_hz"] == (5555.55555556, 50000.0)
    assert before == tuple(
        hashlib.sha256(file.read_bytes()).hexdigest() for file in files
    )
