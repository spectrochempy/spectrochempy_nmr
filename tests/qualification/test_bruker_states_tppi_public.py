"""Qualify the public States-TPPI reader and its FFT presentation."""

from dataclasses import replace
import hashlib
import os
from importlib import import_module
from pathlib import Path

import nmrglue as ng
import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr.experiment import Experiment


@pytest.mark.parametrize("dc", [False, True])
@pytest.mark.parametrize("filtered", [False, True])
def test_public_states_tppi_uses_normalized_dataset(
    monkeypatch, tmp_path, dc, filtered
):
    configured = os.environ.get("NMR_BRUKER_PUBLIC_STATES_TPPI_DATA")
    if not configured:
        pytest.skip("configure the real public States-TPPI reference")
    path = Path(configured)
    files = [
        path / name
        for name in ("ser", "acqus", "acqu2s", "pdata/1/procs", "pdata/1/proc2s")
    ]
    before = [hashlib.sha256(file.read_bytes()).hexdigest() for file in files]
    _check_route(path, monkeypatch, tmp_path, dc, filtered)
    assert [hashlib.sha256(file.read_bytes()).hexdigest() for file in files] == before


def _check_route(path, monkeypatch, tmp_path, dc, filtered):
    if not hasattr(ng.interop, "normalize_bruker_ser_states_tppi"):
        pytest.skip("requires the provider States-TPPI normalization contract")
    options = dict(remove_dc_offset=dc, remove_digital_filter=filtered)
    reader = import_module("spectrochempy_nmr.readers.read_topspin")
    converter = import_module("spectrochempy_nmr.interop")
    with monkeypatch.context() as baseline:
        baseline.setattr(reader, "_read_normalized_ser", lambda *args, **kwargs: None)
        expected = scp.read_topspin(path / "ser", **options)

    captured = []
    contracts = []
    convert = converter.from_interop

    def capture(result, **kwargs):
        dataset = convert(result, **kwargs)
        captured.append(dataset)
        contracts.append(result)
        return dataset

    with monkeypatch.context() as migrated:
        migrated.setattr(converter, "from_interop", capture)
        actual = scp.read_topspin(path / "ser", **options)

    assert len(captured) == 1 and actual is captured[0]
    contract = contracts[0]
    presentation = contract.acquisition["fft_presentation"]
    malformed = replace(
        contract,
        acquisition={
            **contract.acquisition,
            "fft_presentation": {
                **presentation,
                "F1": {**presentation["F1"], "transform": "conjugate-input"},
            },
        },
    )
    with pytest.raises(ValueError, match="unsupported SER FFT presentation transform"):
        convert(malformed, profile="states-tppi")
    assert tuple(actual.c.labels) == ("E1", "E2")
    assert actual.meta.tppi_alternation_state == "encoded"
    assert actual.meta.acquisition["dc_offset"]["applied"] is dc
    assert actual.meta.acquisition["digital_filter"]["applied"] is filtered
    assert actual.meta.acquisition["scan_gain"]["divisor"] == 288
    np.testing.assert_allclose(actual.data.conj(), expected.data, rtol=2e-12, atol=1e-7)
    assert Experiment(actual).process().x.units == scp.ur.Hz
    if not dc and not filtered:
        target = tmp_path / "states-tppi.scp"
        actual.write(target, confirm=False)
        restored = scp.read(target)
        np.testing.assert_array_equal(restored.data, actual.data)
        assert restored.meta.tppi_alternation_state == "encoded"
        assert (
            restored.meta.source_metadata["pulseprogram_raw"].tobytes()
            == (path / "pulseprogram").read_bytes()
        )

    with pytest.raises(NotImplementedError, match="PENDING-F2"):
        actual.fft(dim="y")

    for dim, domains, alternation in (
        ("x", ("time", "frequency"), "encoded"),
        ("y", ("frequency", "frequency"), "applied"),
    ):
        actual, expected = actual.fft(dim=dim), expected.fft(dim=dim)
        assert Experiment(actual).domains == domains
        assert actual.meta.tppi_alternation_state == alternation
        np.testing.assert_allclose(actual.data, expected.data, rtol=2e-12, atol=2e-6)
        assert actual.coordset[dim].units == expected.coordset[dim].units == scp.ur.ppm
        np.testing.assert_allclose(
            actual.coordset[dim].data,
            expected.coordset[dim].data,
            rtol=0,
            atol=0.00101,
        )


@pytest.mark.parametrize("paired_ser", [5], indirect=True)
def test_synthetic_public_states_tppi_asymmetric_branches(
    paired_ser,
    tmp_path,
    monkeypatch,
):
    _check_route(paired_ser, monkeypatch, tmp_path, True, True)
