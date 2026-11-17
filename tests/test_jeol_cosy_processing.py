"""Check the opt-in JEOL COSY processing contract without external datasets."""

from nmrglue import jeol
import numpy as np
import pytest

from spectrochempy_nmr import process_jeol_cosy_magnitude


@pytest.fixture
def cosy_reader(monkeypatch):
    header = {
        "data_format": "two_d",
        "data_axis_type": ["real_complex"] * 2,
        "reversed": [False, True],
        "data_axis_ranged": ["ranged"] * 2,
        "data_points": [16, 8],
        "data_offset_start": [0, 0],
        "data_offset_stop": [15, 7],
        "units": [("none", 1, "second")] * 2,
        "data_axis_start": [0.0, 0.0],
        "data_axis_stop": [15 / 16, 7 / 8],
    }
    parameters = {
        "experiment": "cosy.jxp",
        "x_domain": "1H",
        "y_domain": "1H",
        "x_points": 16,
        "y_points": 8,
        "x_sweep": 16,
        "y_sweep": 8,
        "x_freq": 400e6,
        "y_freq": 400e6,
        "x_offset": 0.5 / 400,
        "y_offset": 0,
    }
    # Reciprocal lines at (0, 3) and (3, 0) Hz, with the second line stronger.
    # The direct carrier cancels the half-dwell grid shift in this fixture.
    t1, t2 = np.indices((8, 16))
    fid = np.exp(2j * np.pi * (3 * t2 / 16))
    fid += 3 * np.exp(2j * np.pi * (-3 * t1 / 8))
    monkeypatch.setattr(
        jeol, "read", lambda path: ({"header": header, "parameters": parameters}, fid)
    )
    return header


def test_magnitude_makes_reciprocal_amplitudes_equal(cosy_reader):
    frequencies = [0.0, 3.0]
    result = process_jeol_cosy_magnitude(
        "cosy.jdf", frequency_grid=(frequencies, frequencies)
    )
    assert result.meta.physical_f1_mode == "indeterminate"
    assert result.data[0, 1] > 0
    assert result.data[0, 1] == pytest.approx(result.data[1, 0])
    expected = (np.sin(np.pi * np.arange(8) / 7) ** 2).sum()
    expected *= (np.sin(np.pi * np.arange(16) / 15) ** 2).sum()
    # Window leakage from the other line contributes to the weaker peak.
    assert 0.8 * expected < result.data[0, 1] < 1.4 * expected


def test_magnitude_rejects_listed_sampling(cosy_reader):
    cosy_reader["data_axis_ranged"][1] = "listed"
    with pytest.raises(ValueError, match="supported two-section"):
        process_jeol_cosy_magnitude("cosy.jdf")


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
        process_jeol_cosy_magnitude("cosy.jdf", frequency_grid=(axis, [1, 2]))
