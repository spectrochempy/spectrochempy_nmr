"""Independent numeric oracles for the external-data qualification metrics."""

import importlib.util
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


def test_external_qualification_refuses_unverified_archive(tmp_path):
    archive = tmp_path / "invalid.zip"
    archive.write_bytes(b"not the published acquisition archive")
    with pytest.raises(ValueError, match="does not match Zenodo"):
        qualification.qualify(archive)
