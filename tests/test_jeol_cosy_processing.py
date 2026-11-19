"""Check the opt-in JEOL COSY processing contract without external datasets."""

import numpy as np
import pytest

from spectrochempy_nmr import process_jeol_cosy_magnitude


from test_jeol_common import qf_jdf as _qf_jdf


@pytest.fixture
def cosy_reader(tmp_path, monkeypatch):
    path, stored, _, _ = _qf_jdf.__wrapped__(tmp_path, monkeypatch)
    t1, t2 = np.indices((8, 16))
    fid = np.exp(2j * np.pi * (3 * t2 / 16))
    fid += 3 * np.exp(2j * np.pi * (-3 * t1 / 8))
    stored.metadata["parameters"].update(x_offset=0.5 / 400, y_offset=0)
    from nmrglue import jeol

    monkeypatch.setattr(jeol, "read", lambda *a, **kw: (stored.metadata, fid.copy()))
    return stored.metadata["header"], path


def test_magnitude_makes_reciprocal_amplitudes_equal(cosy_reader):
    frequencies = [0.0, 3.0]
    result = process_jeol_cosy_magnitude(
        cosy_reader[1], frequency_grid=(frequencies, frequencies)
    )
    assert result.meta.physical_f1_mode == "indeterminate"
    assert result.data[0, 1] > 0
    assert result.data[0, 1] == pytest.approx(result.data[1, 0])
    expected = (np.sin(np.pi * np.arange(8) / 7) ** 2).sum()
    expected *= (np.sin(np.pi * np.arange(16) / 15) ** 2).sum()
    # Window leakage from the other line contributes to the weaker peak.
    assert 0.8 * expected < result.data[0, 1] < 1.4 * expected


def test_magnitude_rejects_listed_sampling(cosy_reader):
    cosy_reader[0]["data_axis_ranged"][1] = "listed"
    with pytest.raises(ValueError, match="supported two-section"):
        process_jeol_cosy_magnitude(cosy_reader[1])


@pytest.mark.parametrize(
    "axis",
    [
        np.array([1 + 2j, 2 + 3j]),
        np.ma.array([1.0, 2.0], mask=[False, True]),
    ],
    ids=["complex", "masked"],
)
def test_magnitude_rejects_nonphysical_coordinates(cosy_reader, axis):
    with pytest.raises(ValueError, match="real, unmasked"):
        process_jeol_cosy_magnitude(cosy_reader[1], frequency_grid=(axis, [1, 2]))
