"""Installed-provider Bruker tests; real corpus remains separately opt-in."""

from copy import deepcopy
from importlib import import_module
from importlib.metadata import distribution
from inspect import signature
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import spectrochempy as scp
from spectrochempy.utils.jsonutils import json_encoder

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
    if "fid_preparation" in normalized.acquisition:
        uncorrected = from_interop(provider.interop.normalize_bruker_fid(raw))
        deferred = Experiment(uncorrected).prepare_fid(
            digital_filter="time" if filtered else None
        )
        np.testing.assert_allclose(deferred.data, source.data, atol=1e-12)


def test_synthetic_public_default_fid_defers_filter_until_experiment(
    provider, tmp_path
):
    if (
        "preserve_storage"
        not in signature(provider.interop.normalize_bruker_fid).parameters
    ):
        pytest.skip("requires the Bruker FID preparation contract")
    (tmp_path / "acqus").write_text(
        "##TITLE= Synthetic Bruker qualification\n"
        "##JCAMPDX= 5.00\n"
        "##$TD= 64\n##$SW_h= 2000\n##$SFO1= 125.00025\n"
        "##$BF1= 125\n##$O1= 250\n##$AQ_mod= 3\n"
        "##$DECIM= 16\n##$DSPFVS= 12\n##$GRPDLY= 2.5\n"
        "##$BYTORDA= 0\n##$DTYPA= 0\n##$NS= 2\n##$RG= 4\n##END=\n"
    )
    words = np.zeros(256, dtype="<i4")
    words[:64] = np.arange(64)
    path = tmp_path / "fid"
    path.write_bytes(words.tobytes())
    raw = scp.nmr.read(path)
    legacy = scp.nmr.read(path, remove_digital_filter=True, remove_dc_offset=True)
    assert raw.meta.acquisition["digital_filter"]["applied"] is False
    assert raw.meta.acquisition["dc_offset"]["applied"] is False
    assert raw.fft().shape == raw.shape
    prepared = Experiment(raw).prepare_fid(digital_filter="time", remove_dc_offset=True)
    np.testing.assert_allclose(prepared.fft().data, legacy.fft().data, atol=1e-12)
    np.testing.assert_array_equal(prepared.x.data, legacy.x.data)
    assert prepared.meta.acquisition["digital_filter"]["applied"] is True
    assert prepared.meta.acquisition["dc_offset"]["applied"] is True

    spectrum = Experiment(raw).process(digital_filter="time")
    trace = spectrum.meta.nmr_processing["scp_processing"]
    assert trace["requested"]["digital_filter"] == "time"
    assert trace["applied"]["digital_filter"]["applied"] is True


def _check_public_fid(provider, path, filtered, dc, monkeypatch, *, supported=True):
    reader = import_module("spectrochempy_nmr.readers.read_topspin")
    converter = import_module("spectrochempy_nmr.interop")

    supports_contract = (
        "preserve_storage"
        in signature(provider.interop.normalize_bruker_fid).parameters
    )

    options = dict(remove_digital_filter=filtered, remove_dc_offset=dc)
    with monkeypatch.context() as legacy:
        legacy.setattr(reader, "_read_normalized_fid", lambda *args, **kwargs: None)
        expected = scp.read_topspin(path / "fid", **options)

    def reject_legacy(*args, **kwargs):
        pytest.fail("the FID must use the nmrglue-ng binary reader")

    original_converter = converter.from_interop
    converted = []

    def record_conversion(result):
        dataset = original_converter(result)
        converted.append(dataset)
        return dataset

    with monkeypatch.context() as migrated:
        if supports_contract and supported:
            migrated.setattr(reader, "_read_raw_data", reject_legacy)
        migrated.setattr(converter, "from_interop", record_conversion)
        actual = scp.read_topspin(path / "fid", **options)

    assert actual is not None and expected is not None
    assert actual.dtype == expected.dtype
    assert actual.units == expected.units
    if not supports_contract or not supported:
        assert not converted
        assert json_encoder(actual.meta) == json_encoder(expected.meta)
        np.testing.assert_array_equal(actual.data, expected.data)
        np.testing.assert_array_equal(actual.x.data, expected.x.data)
        return

    assert len(converted) == 1
    assert actual is converted[0]
    assert actual.meta.acquisition_encoding == tuple(expected.meta.encoding)
    assert actual.meta.rgold == expected.meta.rgold
    assert actual.meta.nsold == expected.meta.nsold
    assert actual.meta.acquisition["digital_filter"]["applied"] is filtered
    assert actual.meta.acquisition["dc_offset"]["applied"] is dc
    assert actual.meta.acquisition["scan_gain"]["divisor"] == (
        expected.meta.rgold[0] * expected.meta.nsold[0]
    )
    np.testing.assert_allclose(
        actual.data, expected.data.conj(), rtol=2e-12, atol=2e-10
    )
    np.testing.assert_allclose(
        actual.x.to("us").data, expected.x.data, rtol=2e-5, atol=0.005
    )

    if expected.meta.sf is not None:
        for actual_fft, expected_fft in (
            (actual.fft(), expected.fft()),
            (actual.fft(ppm=False), expected.fft(ppm=False)),
            (Experiment(actual).process(), Experiment(expected).process()),
        ):
            np.testing.assert_allclose(
                actual_fft.data, expected_fft.data, rtol=1e-10, atol=1e-7
            )
            assert actual_fft.x.units == expected_fft.x.units
            np.testing.assert_allclose(
                actual_fft.x.data, expected_fft.x.data, atol=0.001
            )

    canonical = Experiment(actual).process(reference="acquisition")
    np.testing.assert_allclose(
        canonical.data,
        np.fft.fftshift(np.fft.fft(actual.data, n=canonical.size)),
        rtol=1e-10,
        atol=1e-7,
    )


@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("dc", [False, True])
@pytest.mark.parametrize("supported", [True, False])
def test_synthetic_public_bruker_fid_parity(
    provider, tmp_path, monkeypatch, filtered, dc, supported
):
    path = tmp_path / "bruker" / "1"
    path.mkdir(parents=True)
    (path / "acqus").write_text(
        "##TITLE= Synthetic Bruker acquisition\n"
        f"##$TD= {256 if supported else 128}\n"
        "##$SW= 16\n##$SW_h= 2000\n##$SFO1= 125\n"
        "##$BF1= 125\n##$O1= 0\n##$AQ_mod= 3\n##$DECIM= 16\n"
        "##$DSPFVS= 12\n##$GRPDLY= 2.5\n##$BYTORDA= 1\n##$DTYPA= 0\n"
        "##$NUC1= <13C>\n##$NS= 4\n##$RG= 3\n##END=\n"
    )
    procs = path / "pdata" / "1" / "procs"
    procs.parent.mkdir(parents=True)
    procs.write_text(
        "##TITLE= Synthetic Bruker processing\n"
        "##$SF= 125\n##$PHC0= 21\n##$PHC1= -8\n##END=\n"
    )
    samples = np.arange(256, dtype=np.int32)
    (path / "fid").write_bytes((samples * (samples % 7 - 3)).astype(">i4").tobytes())
    _check_public_fid(provider, path, filtered, dc, monkeypatch, supported=supported)


@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("dc", [False, True])
def test_real_public_bruker_fid_parity(
    provider, acquisition, monkeypatch, filtered, dc
):
    path, _ = acquisition
    _check_public_fid(provider, path, filtered, dc, monkeypatch)


def test_real_deferred_preparation_matches_public_fid(provider, acquisition):
    if "acqus_files" not in signature(provider.interop.read_bruker_fid).parameters:
        pytest.skip("requires the deferred preparation provider contract")
    path, _ = acquisition
    raw = provider.interop.read_bruker_fid(
        path,
        acqus_files=[p for p in (path / "acqu", path / "acqus") if p.is_file()],
        procs_files=list((path / "pdata" / "1").glob("proc*")),
        read_pulseprogram=False,
    )
    normalized = provider.interop.normalize_bruker_fid(raw, preserve_storage=True)
    if "fid_preparation" not in normalized.acquisition:
        pytest.skip("requires the deferred preparation provider contract")
    source = from_interop(normalized)
    options = dict(
        remove_dc_offset=True, digital_filter="time", normalize_scan_gain=True
    )
    prepared = Experiment(source).prepare_fid(**options)
    public = scp.read_topspin(path / "fid", remove_dc_offset=True)
    assert public.meta.normalization_profile == "bruker-fid-1d-complex-v1"
    np.testing.assert_allclose(prepared.data, public.data, rtol=2e-12, atol=2e-10)
    np.testing.assert_allclose(
        prepared.x.to("s").data, public.x.to("s").data, atol=1e-12
    )
    prepared_fft = Experiment(prepared).process()
    public_fft = Experiment(public).process()
    np.testing.assert_allclose(
        prepared_fft.data, public_fft.data, rtol=2e-12, atol=2e-8
    )
    assert prepared_fft.x.units == public_fft.x.units
    if prepared_fft.x.units == scp.ur.Hz:
        # The core rounds the default Hz display differently for s and us inputs.
        prepared_fft = Experiment(prepared).process(reference="acquisition")
        public_fft = Experiment(public).process(reference="acquisition")
    np.testing.assert_allclose(prepared_fft.x.data, public_fft.x.data, atol=0.001)
    np.testing.assert_array_equal(source.data, normalized.data)
    assert not source.meta.acquisition["dc_offset"]["applied"]
    assert (
        len(prepared.meta.applied_operations) == len(source.meta.applied_operations) + 1
    )
    again = Experiment(prepared).prepare_fid(**options)
    np.testing.assert_array_equal(again.data, prepared.data)
    assert again.meta.applied_operations == prepared.meta.applied_operations
    dc = Experiment(source).prepare_fid(remove_dc_offset=True)
    filtered = Experiment(dc).prepare_fid(digital_filter="time")
    staged = Experiment(filtered).prepare_fid(normalize_scan_gain=True)
    np.testing.assert_array_equal(staged.data, prepared.data)
