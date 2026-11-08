"""Installed-provider real processed 1D qualification with independent bytes."""

from copy import deepcopy
from importlib import import_module
from importlib.metadata import distribution
from inspect import signature
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
import pytest
import spectrochempy as scp
from spectrochempy.utils.jsonutils import json_encoder

from spectrochempy_nmr import from_interop
from spectrochempy_nmr.experiment import Experiment

CASES = {
    "topspin": {
        "environment": "NMR_PDATA_TOPSPIN_DATA",
        "size": 16384,
        "nc": -5,
        "frequency": 500.071553859985,
        "offset": 253.0909,
        "width": 250000.0,
        "hashes": {
            "1r": "f3a0a67aca8f3c54a250e70b3089af5d2e879dfbfd27ab2091f088b53ca6247d",
            "1i": "1c760d5385d0e174017e0afe07cee52f4f62e8b185a7decf3895ba6e9d88c224",
            "procs": "7ff305739fe2df2c9dca4576786568a67d80c055a7bd5fc487aa81f467b3f065",
        },
    },
    "tomato": {
        "environment": "NMR_PDATA_TOMATO_DATA",
        "size": 32768,
        "nc": -1,
        "frequency": 500.159950648881,
        "offset": 11.09915,
        "width": 6002.40096038416,
        "hashes": {
            "1r": "c188b3bfea49bb764ef1f30316e82d8f08db7c3477937ab43fa48d0833e236b7",
            "1i": "ca0d589493f0346e113b091eb46c7562adc268afdeadcf7a3b13838a437ca3bd",
            "procs": "0f9ee607f574b6d355b44eb0ec7c2ff2981272385e576f557cf6d19891c2bc39",
        },
    },
}


@pytest.fixture(scope="module")
def provider():
    if not any(os.environ.get(case["environment"]) for case in CASES.values()):
        pytest.skip(
            "opt-in real pdata corpus: set NMR_PDATA_TOMATO_DATA or NMR_PDATA_TOPSPIN_DATA"
        )
    import nmrglue as ng

    dist = distribution("nmrglue-ng")
    candidate = os.environ.get("NMRGLUE_NG_QUALIFICATION_SOURCE")
    location = Path(candidate) / "nmrglue" if candidate else dist.locate_file("nmrglue")
    assert Path(ng.__file__).resolve().is_relative_to(Path(location).resolve())
    return ng


@pytest.fixture(params=CASES)
def acquisition(request, provider):
    case = CASES[request.param]
    value = os.environ.get(case["environment"])
    if not value:
        pytest.skip(f"external original not configured: {case['environment']}")
    root = Path(value)

    def check():
        for name, expected in case["hashes"].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected

    check()
    yield root, case
    check()


@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
def test_original_pdata_import_process_and_persistence(
    acquisition, provider, complex_data, scaled, monkeypatch, tmp_path
):
    path, case = acquisition
    raw = provider.interop.read_bruker_pdata(
        path, all_components=complex_data, scale_data=scaled
    )
    normalized = provider.interop.normalize_bruker_pdata(raw)
    factor = 2.0 ** case["nc"] if scaled else 1
    real = np.frombuffer((path / "1r").read_bytes(), dtype="<i4").astype(float) * factor
    imaginary = (
        np.frombuffer((path / "1i").read_bytes(), dtype="<i4").astype(float) * factor
    )
    assert real.size == case["size"]
    expected = real + 1j * imaginary if complex_data else real
    np.testing.assert_array_equal(normalized.data, expected)
    source = from_interop(normalized)
    before = deepcopy(dict(source.meta))

    def forbidden_fft(*args, **kwargs):
        raise AssertionError("processed spectral input must not FFT again")

    monkeypatch.setattr(scp.NDDataset, "fft", forbidden_fft, raising=False)
    result = Experiment(source).process()
    axis_hz = (
        case["offset"] * case["frequency"]
        - np.arange(case["size"]) * case["width"] / case["size"]
    )
    np.testing.assert_array_equal(result.data, expected)
    assert np.iscomplexobj(result.data) is complex_data
    np.testing.assert_allclose(result.x.data, axis_hz, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(
        result.x.to("ppm").data, axis_hz / case["frequency"], rtol=1e-12, atol=1e-10
    )
    assert Experiment(result).source_kind == "processed_1d"
    assert result.meta.nmr_processing["scp_processing"]["applied"] == {}
    changed = source.copy()
    changed.meta.readonly = False
    changed.origin = "other-format"
    changed.meta.origin = "agilent"
    changed.meta.normalization_profile = "arbitrary"
    changed.meta.source_metadata = {}
    again = Experiment(changed).process()
    np.testing.assert_array_equal(again.data, expected)
    np.testing.assert_array_equal(again.x.data, result.x.data)
    if complex_data:
        phased = Experiment(source).process(phase="manual", phc0=25)
        np.testing.assert_array_equal(phased.x.data, result.x.data)
        assert not np.array_equal(phased.data, expected)
    else:
        with pytest.raises(ValueError, match="imaginary"):
            Experiment(source).process(phase="manual", phc0=25)
    target = tmp_path / "processed.scp"
    result.write(target, confirm=False)
    restored = scp.read(target)
    np.testing.assert_array_equal(restored.data, expected)
    np.testing.assert_array_equal(restored.x.data, result.x.data)
    np.testing.assert_allclose(
        restored.x.to("ppm").data, axis_hz / case["frequency"], rtol=1e-12, atol=1e-10
    )
    assert restored.meta.source_metadata["procs"]["SF"] == case["frequency"]
    assert restored.meta.interop_axes[0]["reference"]["source"] == "processing"
    assert json.loads(json.dumps(restored.meta.nmr_processing)) == json.loads(
        json.dumps(result.meta.nmr_processing)
    )
    assert dict(source.meta) == before


@pytest.mark.parametrize("proc_axis", [False, True])
@pytest.mark.parametrize("complex_data", [False, True])
def test_public_read_topspin_pdata_parity(
    acquisition, provider, complex_data, proc_axis, monkeypatch, tmp_path
):
    path, case = acquisition
    if not complex_data:
        experiment = tmp_path / "bruker" / "1"
        real_only = experiment / "pdata" / "1"
        real_only.mkdir(parents=True)
        shutil.copy2(path.parents[1] / "acqus", experiment / "acqus")
        for name in ("1r", "procs"):
            shutil.copy2(path / name, real_only / name)
        path = real_only
    component = path / ("1i" if complex_data else "1r")
    reader = import_module("spectrochempy_nmr.readers.read_topspin")

    def unsupported(*args, **kwargs):
        raise ValueError("unqualified profile")

    with monkeypatch.context() as legacy:
        legacy.setattr(provider.interop, "read_bruker_pdata", unsupported)
        expected = scp.read_topspin(component, proc_axis=proc_axis)

    def reject_legacy(*args, **kwargs):
        pytest.fail("qualified 1D pdata should not use the vendored payload decoder")

    converter = import_module("spectrochempy_nmr.interop")
    converted = []

    def record_conversion(result):
        dataset = from_interop(result)
        converted.append(dataset)
        return dataset

    with monkeypatch.context() as migrated:
        migrated.setattr(reader, "read_pdata", reject_legacy)
        migrated.setattr(converter, "from_interop", record_conversion)
        actual = scp.read_topspin(component, proc_axis=proc_axis)

    assert actual is not None and expected is not None
    assert np.iscomplexobj(actual.data) is complex_data
    assert actual.dtype == expected.dtype
    assert actual.dims == expected.dims
    assert actual.units == expected.units
    assert actual.origin == expected.origin
    assert actual.name == expected.name
    np.testing.assert_array_equal(actual.data, expected.data)
    assert actual.x.units == expected.x.units
    assert actual.x.title == expected.x.title
    if (
        "axis_reference"
        not in signature(provider.interop.normalize_bruker_pdata).parameters
    ):
        assert json_encoder(actual.meta) == json_encoder(expected.meta)
        np.testing.assert_array_equal(actual.x.data, expected.x.data)
        assert json_encoder(actual.x.meta) == json_encoder(expected.x.meta)
        return

    assert len(converted) == 1 and actual is converted[0]
    assert actual.x.meta.acquisition_frequency == expected.x.meta.acquisition_frequency
    sfo = expected.x.meta.acquisition_frequency.to("MHz").magnitude
    sf = expected.meta.sf[-1].to("MHz").magnitude
    assert sfo != sf
    frequency = sf if proc_axis else sfo
    assert actual.x.meta.reference_frequency == frequency * scp.ur.MHz
    np.testing.assert_allclose(actual.x.to("Hz").data, actual.x.data * frequency)
    assert actual.x.meta.reference_source == (
        "processing" if proc_axis else "acquisition"
    )
    np.testing.assert_allclose(actual.x.data, expected.x.data, rtol=0, atol=0.00051)
    assert actual.meta.acquisition["storage_scaling"]["NC_proc"] == case["nc"]
    assert actual.meta.applied_operations[0]["receiver_gain"] == expected.meta.rgold[-1]
    assert actual.meta.applied_operations[0]["scale"] == 1 / expected.meta.rgold[-1]
    processed = Experiment(actual).process()
    np.testing.assert_array_equal(processed.data, expected.data)
    np.testing.assert_array_equal(processed.x.data, actual.x.data)
    assert processed.meta.nmr_processing["scp_processing"]["applied"] == {}
    actual_meta, expected_meta = (
        json_encoder(actual.meta)["data"],
        json_encoder(expected.meta)["data"],
    )
    assert {key: actual_meta[key] for key in expected_meta} == expected_meta


def test_public_read_topspin_falls_back_for_degenerate_calibration(
    acquisition, provider, monkeypatch, tmp_path
):
    original, _ = acquisition
    experiment = tmp_path / "bruker" / "1"
    path = experiment / "pdata" / "1"
    shutil.copytree(original, path)
    shutil.copy2(original.parents[1] / "acqus", experiment / "acqus")
    with monkeypatch.context() as legacy:

        def unsupported(*args, **kwargs):
            raise ValueError("unqualified profile")

        legacy.setattr(provider.interop, "read_bruker_pdata", unsupported)
        original_data = scp.read_topspin(path / "1r")
    procs = path / "procs"
    procs.write_bytes(
        re.sub(rb"(?m)^##\$OFFSET=[^\r\n]*", b"##$OFFSET= 1e20", procs.read_bytes())
    )
    reader = import_module("spectrochempy_nmr.readers.read_topspin")
    legacy_read = reader.read_pdata
    calls = []

    def record_legacy(*args, **kwargs):
        calls.append(args[0])
        return legacy_read(*args, **kwargs)

    monkeypatch.setattr(reader, "read_pdata", record_legacy)
    result = scp.read_topspin(path / "1r")
    assert calls == [path]
    np.testing.assert_array_equal(result.data, original_data.data)
    np.testing.assert_array_equal(result.x.data, original_data.x.data)
