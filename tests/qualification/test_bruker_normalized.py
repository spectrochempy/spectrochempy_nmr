"""Installed-provider Bruker tests; real corpus remains separately opt-in."""

from copy import deepcopy
from importlib.metadata import distribution
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import from_interop
from spectrochempy_nmr.experiment import Experiment

CASES = {
    "legacy": {
        "env": "NMR_BRUKER_LEGACY_DATA",
        "points": 2048,
        "storage": 2048,
        "sw": 10000.0,
        "ref": 600.13,
        "carrier": 2820.61100006104,
        "size": 4096,
        "delay": 71.625,
        "skip": 73,
        "hashes": (
            "fc52cb12395e357efa0990cbed9298e9bdd90c2937ccc4ccd902aa99aef46d14",
            "1f7cc2d3a62a3cf80ce103209cff1cbcb3a6c839df1a3e321e84a33508547800",
        ),
    },
    "topspin": {
        "env": "NMR_BRUKER_TOPSPIN_DATA",
        "points": 12487,
        "storage": 12544,
        "sw": 250000.0,
        "ref": 500.07,
        "carrier": 3117.4,
        "size": 32768,
        "delay": 76.0,
        "skip": 78,
        "hashes": (
            "a471ce0ec0c5f28b02c1ae05200f1b0300e29f042d36c3d5ff2d53c433219ed6",
            "00f86298b13a4f9a116feba96b9691fde6bf163ca4d1291a84d4ac8a82d12560",
        ),
    },
}


@pytest.fixture(scope="module")
def provider():
    ng = pytest.importorskip("nmrglue", reason="optional external nmrglue-ng provider")
    dist = distribution("nmrglue-ng")
    assert (
        Path(ng.__file__)
        .resolve()
        .is_relative_to(Path(dist.locate_file("nmrglue")).resolve())
    )
    assert hasattr(
        ng.interop, "normalize_bruker_fid"
    ), "the qualified provider is required"
    return ng


def _hashes(path):
    return tuple(
        hashlib.sha256((path / name).read_bytes()).hexdigest()
        for name in ("fid", "acqus")
    )


@pytest.fixture(params=CASES)
def acquisition(request):
    case = CASES[request.param]
    location = os.environ.get(case["env"])
    if not location:
        pytest.skip(f"local real corpus: set {case['env']}")
    path = Path(location)
    assert _hashes(path) == case["hashes"]
    yield path, case
    assert _hashes(path) == case["hashes"], "original acquisition changed"


def _binary_oracle(path, case):
    dtype = ">i4" if case is CASES["legacy"] else "<i4"
    words = np.frombuffer((path / "fid").read_bytes(), dtype=dtype)
    assert len(words) == case["storage"] * 2
    raw = words[::2].astype(float) + 1j * words[1::2]
    assert np.count_nonzero(raw[case["points"] :]) == 0
    return raw


@pytest.mark.parametrize("filtered", [False, True])
def test_real_normalized_experiment_axes_samples_and_persistence(
    provider, acquisition, tmp_path, filtered
):
    path, case = acquisition
    raw = provider.interop.read_bruker_fid(path)
    normalized = provider.interop.normalize_bruker_fid(
        raw, remove_digital_filter=filtered
    )
    expected = _binary_oracle(path, case)[: case["points"]].copy()
    if filtered:
        expected = np.roll(expected, -int(np.floor(case["delay"])))
        add = max(case["skip"] - 6, 0)
        expected[:add] += expected[-1 : -add - 1 : -1].copy()
        expected = expected[: -case["skip"]]
    expected = np.conj(expected)
    np.testing.assert_allclose(normalized.data, expected, rtol=1e-12, atol=1e-7)
    source = from_interop(normalized)
    assert Experiment(source).is_time_domain
    source_meta = deepcopy(dict(source.meta))
    unreferenced = Experiment(source).process(size=case["size"])
    assert unreferenced.x.units == scp.ur.Hz
    result = Experiment(source).process(size=case["size"], reference="acquisition")
    np.testing.assert_allclose(
        result.data,
        np.fft.fftshift(np.fft.fft(expected, n=case["size"])),
        rtol=1e-10,
        atol=1e-6,
    )
    axis = (
        case["carrier"]
        - (np.arange(case["size"]) - case["size"] // 2) * case["sw"] / case["size"]
    ) / case["ref"]
    np.testing.assert_allclose(result.x.data, axis, rtol=1e-10, atol=1e-10)
    assert Experiment(result).is_frequency_domain
    changed = source.copy()
    changed.meta.readonly = False
    changed.origin = "unknown"
    changed.meta.origin = "agilent"
    changed.meta.normalization_profile = "opaque"
    changed.meta.source_metadata = {}
    other = Experiment(changed).process(size=case["size"], reference="acquisition")
    np.testing.assert_array_equal(result.data, other.data)
    np.testing.assert_array_equal(result.x.data, other.x.data)
    target = tmp_path / "normalized.scp"
    result.write(target, confirm=False)
    restored = scp.read(target)
    np.testing.assert_array_equal(restored.data, result.data)
    np.testing.assert_allclose(restored.x.data, axis, rtol=1e-10, atol=1e-10)
    assert json.loads(json.dumps(restored.meta.nmr_processing)) == json.loads(
        json.dumps(result.meta.nmr_processing)
    )
    assert restored.meta.acquisition["digital_filter"]["applied"] is filtered
    assert restored.meta.source_metadata["acqus"]["NS"] == raw.metadata["acqus"]["NS"]
    assert restored.meta.source_metadata["acqus"]["RG"] == raw.metadata["acqus"]["RG"]
    assert dict(source.meta) == source_meta
    np.testing.assert_array_equal(raw.data, _binary_oracle(path, case))


@pytest.mark.parametrize("filtered", [False, True])
def test_synthetic_provider_binary_to_experiment_chain(provider, tmp_path, filtered):
    acqus = """##TITLE= Synthetic Bruker qualification
##JCAMPDX= 5.00
##$TD= 64
##$SW_h= 2000
##$SFO1= 125.00025
##$BF1= 125
##$O1= 250
##$AQ_mod= 3
##$DECIM= 16
##$DSPFVS= 12
##$GRPDLY= 2.5
##$BYTORDA= 0
##$DTYPA= 0
##$NUC1= <13C>
##END=
"""
    (tmp_path / "acqus").write_text(acqus)
    values = np.arange(32)
    samples = values + 1j * (values + 1)
    words = np.zeros(256, dtype="<i4")
    words[:64:2] = values
    words[1:64:2] = values + 1
    (tmp_path / "fid").write_bytes(words.tobytes())
    raw = provider.interop.read_bruker_fid(tmp_path, read_pulseprogram=False)
    normalized = provider.interop.normalize_bruker_fid(
        raw, remove_digital_filter=filtered
    )
    expected = np.roll(samples, -2)[:-4] if filtered else samples
    source = from_interop(normalized)
    result = Experiment(source).process(size=64, reference="acquisition")
    np.testing.assert_allclose(
        result.data,
        np.fft.fftshift(np.fft.fft(np.conj(expected), n=64)),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result.x.data, (250 - (np.arange(64) - 32) * 2000 / 64) / 125, atol=1e-12
    )
    assert source.meta.acquisition["trimmed_padding_points"] == 96
