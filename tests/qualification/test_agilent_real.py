"""Opt-in real-provider qualification; no synthetic nmrglue module is used.

Set NMR_AGILENT_REAL_DATA to the downloaded nmrML corpus directory. A configured
but missing/mismatched corpus or provider is an error, never a silent skip.
NMR_AGILENT_LEGACY_DATA optionally adds an existing local historical corpus.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
from copy import deepcopy

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import from_interop, read_agilent_interop

HOP = "FAM013_AHTM.PROTON_04.fid"
CASES = {
    HOP: (
        "ffdb15ed7ec30d3e01d86a8a743958ea6e165c618f791a728956bee43a06ce81",
        "458e0230178a7642effa116ff32f005edda092f698899998300a4620096ca370",
        32768,
        12019.2307692,
        599.8311617,
    ),
    "agilent_1d": (
        "4a4ccf9b8a8c20e4dd5d151564d84585fd13a15f53cc967cbaa77c556b29e145",
        "423d25e161adc5b5ad5a504ece04667c89835f698b4ba19ddd0a2de545bd2fbb",
        1500,
        50000.0,
        125.6811107,
    ),
}


@pytest.fixture(scope="module")
def provider():
    if not os.environ.get("NMR_AGILENT_REAL_DATA"):
        pytest.skip("opt-in real corpus: set NMR_AGILENT_REAL_DATA")
    import nmrglue as ng
    from importlib.metadata import distribution

    dist = distribution("nmrglue-ng")
    assert (
        Path(ng.__file__)
        .resolve()
        .is_relative_to(Path(dist.locate_file("nmrglue")).resolve())
    ), "qualification must use the installed external provider"
    source = Path(os.environ["NMR_AGILENT_REAL_DATA"]) / HOP
    for name, expected in zip(("fid", "procpar"), CASES[HOP][:2]):
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected
    return ng


@pytest.fixture(params=[HOP, "agilent_1d"])
def acquisition(request, provider):
    if request.param == HOP:
        root = Path(os.environ["NMR_AGILENT_REAL_DATA"])
    else:
        local = os.environ.get("NMR_AGILENT_LEGACY_DATA")
        if not local:
            pytest.skip("optional existing historical corpus not configured")
        root = Path(local)
    path = root / request.param
    for name, expected in zip(("fid", "procpar"), CASES[request.param][:2]):
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == expected
    return path


def _binary_oracle(path):
    """Decode only these single-trace 1D files, without either reader library."""
    payload = (path / "fid").read_bytes()
    (
        blocks,
        traces,
        points,
        width,
        trace_bytes,
        block_bytes,
        _,
        status,
        headers,
    ) = struct.unpack(">6i2hi", payload[:32])
    assert blocks == traces == 1 and points % 2 == 0
    # These original files omit S_COMPLEX despite storing interleaved samples.
    # This oracle is deliberately restricted to their observed float headers.
    assert status == 201
    assert trace_bytes == points * width
    assert len(payload) == 32 + blocks * block_bytes
    dtype = ">f4" if status & 8 else (">i4" if status & 4 else ">i2")
    assert np.dtype(dtype).itemsize == width
    values = np.frombuffer(
        payload, dtype=dtype, count=points, offset=32 + 28 * headers
    ).astype(np.float32)
    return values[::2] + 1j * values[1::2]


def test_real_chain_samples_axes_metadata_and_ownership(acquisition, provider):
    before = {name: (acquisition / name).read_bytes() for name in ("fid", "procpar")}
    raw = provider.interop.read_agilent_fid(str(acquisition))
    normalized = provider.interop.normalize_agilent_fid(raw)
    converted = from_interop(normalized)
    facade = read_agilent_interop(acquisition / "fid")
    _, _, points, sw, frequency = CASES[acquisition.name]
    oracle = _binary_oracle(acquisition)
    for result in (raw, normalized, converted, facade):
        assert result.data.shape == (points,)
        assert result.data.dtype == np.complex64
        np.testing.assert_array_equal(result.data, oracle)
    np.testing.assert_allclose(
        facade.x.data, np.arange(points) / sw, rtol=1e-10, atol=1e-14
    )
    assert facade.x.units == scp.ur.s
    assert facade.meta.axis_roles == ("F2",)
    assert facade.meta.interop_domains == ("time",)
    assert facade.meta.isfreq == [False]
    assert facade.meta.normalization_profile == "agilent-fid-1d-complex-v1"
    assert facade.meta.normalization_state == "storage-normalized"
    assert facade.meta.acquisition["observation_frequency_mhz"] == frequency
    assert facade.meta.acquisition["dwell_time_s"] == 1 / sw
    assert facade.meta.acquisition["carrier_ppm"] is None
    assert facade.meta.acquisition["original_reference"] == dict(
        normalized.acquisition["original_reference"]
    )
    assert facade.meta.provenance["source"] == str(acquisition.resolve())
    assert facade.meta.source_metadata["procpar"] == raw.metadata["procpar"]
    operations = facade.meta.applied_operations
    assert len(operations) == 1
    assert operations[0]["scale"] is None
    assert operations[0]["conjugation"] == "none"
    converted.data[0] = 0
    converted.x.data[0] = 123
    np.testing.assert_array_equal(normalized.data, oracle)
    assert normalized.axes[0].values[0] == 0
    for name, payload in before.items():
        assert (acquisition / name).read_bytes() == payload


def test_real_legacy_parity_without_claiming_scan_normalization(acquisition, provider):
    result = read_agilent_interop(acquisition)
    legacy = scp.read_agilent(acquisition)
    np.testing.assert_array_equal(result.data, legacy.data.squeeze())
    # Convert numeric values rather than creating another rounded Coord in s.
    np.testing.assert_allclose(
        result.x.data,
        (legacy.x.data * legacy.x.units).to("s").magnitude,
        rtol=1e-10,
        atol=1e-9,
    )
    assert result.meta.acquisition["NS"] is None
    assert result.meta.acquisition["NE"] is None
    assert legacy.meta.ns == legacy.meta.ne == 1


def _assert_real_scan_counts(dataset, acquisition):
    expected = {HOP: 160, "agilent_1d": 512}[acquisition.name]
    scale, _, _, _, ctcount, _, _, _, _ = struct.unpack(
        ">4hi4f", (acquisition / "fid").read_bytes()[32:60]
    )
    assert ctcount == expected and scale == 0
    counts = dataset.meta.acquisition["scan_counts"]
    # SCP serialization preserves values but represents tuples as JSON lists.
    assert json.loads(json.dumps(counts)) == {
        "requested": expected,
        "procpar_completed": expected,
        "block_completed": [expected],
        "block_scales": [0],
        "completed": expected,
        "source": "fid.blockheader.ctcount",
        "status": "consistent",
        "invalid": [],
    }
    for key in ("nt", "ct"):
        assert dataset.meta.source_metadata["procpar"][key]["values"] == [str(expected)]
    assert dataset.meta.acquisition["NS"] is None
    assert dataset.meta.acquisition["NE"] is None
    assert dataset.meta.acquisition["normalization"] == "none"
    assert dataset.meta.applied_operations[0]["scale"] is None


def test_real_scan_count_evidence_preserved_without_scaling(acquisition, provider):
    result = read_agilent_interop(acquisition)
    _assert_real_scan_counts(result, acquisition)
    np.testing.assert_array_equal(result.data, _binary_oracle(acquisition))


def test_real_persistence(acquisition, provider, tmp_path):
    result = read_agilent_interop(acquisition)
    _assert_real_scan_counts(result, acquisition)
    target = tmp_path / "real-fid.scp"
    result.write(target, confirm=False)
    restored = scp.read(target)
    _assert_real_scan_counts(restored, acquisition)
    np.testing.assert_array_equal(restored.data, result.data)
    np.testing.assert_array_equal(restored.x.data, result.x.data)
    for field in ("acquisition", "provenance", "source_metadata", "interop_axes"):
        # SCP serialization represents tuples as JSON lists.
        assert json.loads(json.dumps(getattr(restored.meta, field))) == json.loads(
            json.dumps(getattr(result.meta, field))
        )


def test_real_compatible_copies_stack_and_incompatible_acquisitions_refuse(
    provider, tmp_path
):
    source = Path(os.environ["NMR_AGILENT_REAL_DATA"]) / HOP
    copies = [shutil.copytree(source, tmp_path / name) for name in ("a", "b")]
    stacked = read_agilent_interop(*copies, merge=True)
    assert stacked.shape == (2, 32768)
    assert stacked.meta.series_axis_role == "acquisition_index"
    assert len(stacked.meta.individual_provenance) == 2
    assert [
        item["scan_counts"]["completed"] for item in stacked.meta.individual_acquisition
    ] == [160, 160]
    assert stacked.meta.acquisition_order == tuple(str(path) for path in copies)
    np.testing.assert_array_equal(stacked.data[0], stacked.data[1])
    local = os.environ.get("NMR_AGILENT_LEGACY_DATA")
    if local:
        with pytest.raises(ValueError, match="shape|spectral"):
            read_agilent_interop(source, Path(local) / "agilent_1d", merge=True)


def test_controlled_scan_count_variant_is_not_a_new_real_acquisition(
    provider, tmp_path
):
    source = Path(os.environ["NMR_AGILENT_REAL_DATA"]) / HOP
    variant = shutil.copytree(source, tmp_path / "scan-count-variant")
    procpar = provider.varian.read_procpar(str(variant / "procpar"))
    for name, count in (("ns", "4"), ("ne", "2")):
        procpar[name] = deepcopy(procpar["nt"])
        procpar[name]["name"] = name
        procpar[name]["values"] = [count]
    provider.varian.write_procpar(str(variant / "procpar"), procpar, overwrite=True)
    result = read_agilent_interop(variant)
    legacy = scp.read_agilent(variant)
    np.testing.assert_array_equal(result.data, _binary_oracle(source))
    np.testing.assert_array_equal(legacy.data, result.data / 8)
    assert result.meta.acquisition["NS"] == 4
    assert result.meta.acquisition["NE"] == 2
    assert result.meta.applied_operations[0]["scale"] is None


def test_observed_nucleus_uses_transmitter_not_decoupler(provider):
    source = Path(os.environ["NMR_AGILENT_REAL_DATA"]) / HOP
    result = read_agilent_interop(source)
    assert result.meta.source_metadata["procpar"]["tn"]["values"] == ["H1"]
    assert result.meta.source_metadata["procpar"]["dn"]["values"] == ["C13"]
    assert result.meta.acquisition["nucleus"] == "H1"
    assert result.x.title == "H1"
