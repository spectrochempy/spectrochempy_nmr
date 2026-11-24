"""Independent numeric oracles for the external-data qualification metrics."""

import importlib.util
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest


path = Path(__file__).parents[1] / "tools/qualify_distribution_baseline.py"
spec = importlib.util.spec_from_file_location("distribution_qualification", path)
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("sign", [-1, 1])
def test_recovery_uses_signed_physical_area_without_renormalizing(sign, descending):
    x = np.linspace(0, 2, 101)
    injected = sign * (2 + x)
    increment = injected * 0.75
    if descending:
        x, injected, increment = x[::-1], injected[::-1], increment[::-1]
    result = qualification.recovery_metrics(x, injected, increment, (2, 0))
    assert result["expected_area"] == pytest.approx(sign * 6)
    assert result["recovered_area"] == pytest.approx(sign * 4.5)
    assert result["area_bias_percent"] == pytest.approx(-25)


def test_integration_rejects_empty_or_underresolved_window():
    with pytest.raises(ValueError, match="three samples"):
        qualification.area(np.arange(4), np.ones(4), (0.1, 0.2))


def test_recovery_rejects_zero_reference():
    with pytest.raises(ValueError, match="nonzero"):
        qualification.recovery_metrics(np.arange(4), np.zeros(4), np.ones(4), (0, 3))


def test_export_values_are_normalized_to_the_exported_reference():
    result = qualification.regions_from_export(
        "1 8.8 8.5 0.5\n2 7.6 7.4 1.0\n", integrals=True
    )
    assert result == [(8.8, 8.5, 1.0), (7.6, 7.4, 2.0)]


@pytest.mark.parametrize("header", ["A 1.0 # regions", "P 0"])
def test_exported_bounds_without_values_do_not_claim_known_areas(header):
    result = qualification.regions_from_export(
        header + "\n# low high bias slope\n8.8 8.5 0 0\n", integrals=False
    )
    assert result == [(8.8, 8.5, None)]


def test_external_qualification_refuses_unverified_archive(tmp_path):
    archive = tmp_path / "invalid.zip"
    archive.write_bytes(b"not the published acquisition archive")
    with pytest.raises(ValueError, match="does not match Zenodo"):
        qualification.qualify(archive)


@pytest.mark.parametrize("hardlink", [False, True])
def test_cli_cannot_overwrite_input_through_the_same_path_or_a_hardlink(
    tmp_path, monkeypatch, capsys, hardlink
):
    archive = tmp_path / "input.zip"
    archive.write_bytes(b"preserve this source archive")
    output = archive
    if hardlink:
        output = tmp_path / "output.zip"
        output.hardlink_to(archive)
    monkeypatch.setattr(sys, "argv", [str(path), str(archive), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(path), run_name="__main__")
    assert error.value.code == 2
    assert "must not replace the source archive" in capsys.readouterr().err
    assert archive.read_bytes() == b"preserve this source archive"
