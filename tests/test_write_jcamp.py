import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import spectrochempy as scp
from spectrochempy import ur

FREQUENCY = 399.78219837825
INTENSITIES = [0.125, -0.375, 1.2345678901234567, 1.25e-18]


def spectrum(axis=(9.125, 8.875, 8.625, 8.375), unit="ppm"):
    data = scp.NDDataset(INTENSITIES, name="Test 1H spectrum")
    data.set_coordset(scp.Coord(list(axis), units=unit))
    return data


@pytest.mark.parametrize(
    ("axis", "unit"),
    [([9.125, 8.875, 8.625, 8.375], "ppm"), ([-1200, -800, -400, 0], "Hz")],
)
def test_write_jcamp_roundtrip(tmp_path, axis, unit):
    data = spectrum(axis, unit)
    path = tmp_path / "spectrum.jdx"
    assert (
        scp.nmr.write_jcamp(
            data, path, observation_frequency=FREQUENCY * ur.MHz, nucleus="^1H"
        )
        == path
    )
    assert "0.00000000000000000125" in path.read_text(encoding="ascii")
    for restored in (
        scp.read_jcamp(path),
        scp.read_jcamp({path.name: path.read_bytes()}),
    ):
        np.testing.assert_array_equal(restored.data, INTENSITIES)
        np.testing.assert_array_equal(restored.x.data, axis)
        assert restored.x.units == unit
        assert restored.units is None
        assert restored.meta.data_domain == "frequency"
        assert restored.x.meta["acquisition_frequency"] == FREQUENCY * ur.MHz
        assert restored.meta.source_metadata[".OBSERVENUCLEUS"] == ["^1H"]

    exported = scp.nmr.write_jcamp(restored, tmp_path / "second.dx")
    np.testing.assert_array_equal(scp.read_jcamp(exported).data, INTENSITIES)
    np.testing.assert_array_equal(data.data, INTENSITIES)
    np.testing.assert_array_equal(data.x.data, axis)


@pytest.mark.parametrize(
    "invalid", ["irregular", "axis_units", "intensity_units", "mask", "complex", "time"]
)
def test_write_jcamp_rejects_invalid_spectrum(tmp_path, invalid):
    data = spectrum()
    if invalid == "irregular":
        data = spectrum([9, 8, 6, 5])
    elif invalid == "axis_units":
        data = spectrum(unit="s")
    elif invalid == "intensity_units":
        data.units = "V"
    elif invalid == "mask":
        data.mask = [True, False, False, False]
    elif invalid == "complex":
        data = scp.NDDataset([1j, 2j, 3j, 4j], coordset=[data.x])
    elif invalid == "time":
        data.meta.data_domain = "time"
    path = tmp_path / "spectrum.jdx"
    with pytest.raises(ValueError):
        scp.nmr.write_jcamp(data, path, observation_frequency=FREQUENCY, nucleus="^1H")
    assert not path.exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"nucleus": "^1H"}, "frequency"),
        ({"observation_frequency": FREQUENCY}, "nucleus"),
        ({"observation_frequency": -1, "nucleus": "^1H"}, "frequency"),
        ({"observation_frequency": FREQUENCY, "nucleus": "^1H\n##END="}, "nucleus"),
    ],
)
def test_write_jcamp_rejects_invalid_observation(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        scp.nmr.write_jcamp(spectrum(), tmp_path / "spectrum.jdx", **kwargs)


def test_write_jcamp_overwrite_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "existing.jdx"
    path.write_bytes(b"existing spectrum")
    kwargs = {"observation_frequency": FREQUENCY, "nucleus": "^1H"}
    with pytest.raises(FileExistsError):
        scp.nmr.write_jcamp(spectrum(), path, **kwargs)
    assert path.read_bytes() == b"existing spectrum"

    def fail_replace(_self, _target):
        raise OSError("simulated replacement failure")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError, match="replacement failure"):
            scp.nmr.write_jcamp(spectrum(), path, overwrite=True, **kwargs)
    assert path.read_bytes() == b"existing spectrum"
    assert list(tmp_path.iterdir()) == [path]

    scp.nmr.write_jcamp(spectrum(), path, overwrite=True, **kwargs)
    np.testing.assert_array_equal(scp.read_jcamp(path).data, INTENSITIES)


def test_write_jcamp_epicatechin_roundtrip(tmp_path):
    fixture = os.environ.get("NMR_JCAMP_EPICATECHIN_DATA")
    if fixture is None:
        pytest.skip("set NMR_JCAMP_EPICATECHIN_DATA to the qualified JCAMP file")
    source = Path(fixture).resolve()
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "3047748b1d286a03c374c0975650c71f80f065d10c71d780cde40c6ac265dd8f"
    )
    data = scp.read_jcamp(source)
    restored = scp.read_jcamp(scp.nmr.write_jcamp(data, tmp_path / "epicatechin.jdx"))
    assert restored.size == 104858
    np.testing.assert_array_equal(restored.data, data.data)
    np.testing.assert_array_equal(restored.x.data, data.x.data)
    assert restored.x.units == "Hz"
    for key in ("FIRSTX", "LASTX"):
        assert float(restored.meta.source_metadata[key][0]) == float(
            data.meta.source_metadata[key][0]
        )
