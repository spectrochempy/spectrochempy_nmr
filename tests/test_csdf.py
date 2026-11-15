import base64
import json

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr._csdf import _read_csdf


@pytest.fixture
def spectrum():
    values = np.arange(6, dtype="<f4").reshape(2, 3)
    values = (values + 1j * (values + 10)).astype("<c8")
    model = {
        "version": "1.0",
        "dimensions": [
            {
                "type": "linear",
                "count": 3,
                "increment": "-50 Hz",
                "coordinates_offset": "0.2 kHz",
                "origin_offset": "4e-10 (EHz) ",
                "label": "F2",
            },
            {
                "type": "linear",
                "count": 2,
                "increment": "25 Hz",
                "origin_offset": "100 MHz",
                "label": "F1",
            },
        ],
        "dependent_variables": [
            {
                "type": "internal",
                "quantity_type": "scalar",
                "encoding": "base64",
                "numeric_type": "complex64",
                "components": [base64.b64encode(values.tobytes()).decode("ascii")],
                "unit": "V",
            },
        ],
    }
    return model, values


def test_csdf_preserves_numpy_order_and_frequency_coordinates(tmp_path, spectrum):
    model, expected = spectrum
    path = tmp_path / "spectrum.csdf"
    path.write_text(json.dumps({"csdm": model}))

    values, (f1, f2), unit = _read_csdf(path)

    np.testing.assert_array_equal(values, expected)
    assert values.dtype == expected.dtype
    np.testing.assert_array_equal(f1.values_hz, [0, 25])
    np.testing.assert_array_equal(f2.values_hz, [200, 150, 100])
    assert (f1.origin_offset_hz, f2.origin_offset_hz) == (100e6, 400e6)
    assert unit == "V"


def test_csdf_rejects_payload_shape_mismatch(tmp_path, spectrum):
    model, _ = spectrum
    model["dimensions"][0]["count"] = 2
    path = tmp_path / "spectrum.csdf"
    path.write_text(json.dumps({"csdm": model}))

    with pytest.raises(ValueError, match="payload size"):
        _read_csdf(path)


def test_csdf_public_reader_preserves_coordinates_and_origins(tmp_path, spectrum):
    model, expected = spectrum
    model["dimensions"][0]["coordinates_offset"] = "60718.26753 Hz"
    path = tmp_path / "spectrum.csdf"
    path.write_text(json.dumps({"csdm": model}))

    direct = scp.nmr.read_csdf(path)
    for dataset in (direct, scp.nmr.read(path), scp.nmr.read(path, protocol="csdf")):
        np.testing.assert_array_equal(dataset.data, expected)
        assert dataset.dtype == expected.dtype
        assert dataset.units == scp.ur.V
        np.testing.assert_array_equal(dataset.y.data, [0, 25])
        np.testing.assert_allclose(
            dataset.x.data,
            60718.26753 - 50 * np.arange(3),
            rtol=0,
            atol=1e-9,
        )
        assert dataset.x.units == dataset.y.units == scp.ur.Hz
        assert (dataset.y.title, dataset.x.title) == ("F1", "F2")
        assert dataset.y.meta.csdf_origin_offset_hz == 100e6
        assert dataset.x.meta.csdf_origin_offset_hz == 400e6

    model["dimensions"][0]["count"] = 2
    path.write_text(json.dumps({"csdm": model}))
    with pytest.raises(ValueError, match="payload size"):
        scp.nmr.read(path)


def test_csdf_export_preserves_centered_complex_spectrum(tmp_path, spectrum):
    model, expected = spectrum
    model["dimensions"][0]["complex_fft"] = True
    source = tmp_path / "source.csdf"
    source.write_text(json.dumps({"csdm": model}))
    dataset = scp.nmr.read_csdf(source)
    np.testing.assert_array_equal(dataset.x.data, [250, 200, 150])
    assert scp.nmr.Experiment(dataset).is_processed

    target = tmp_path / "export.csdf"
    assert scp.nmr.write_csdf(dataset, target) == target
    restored = scp.nmr.read_csdf(target)
    np.testing.assert_array_equal(restored.data, expected)
    assert restored.dtype == expected.dtype
    assert restored.units == dataset.units
    np.testing.assert_array_equal(restored.x.data, dataset.x.data)
    np.testing.assert_array_equal(restored.y.data, dataset.y.data)
    assert restored.y.meta.csdf_origin_offset_hz == 100e6
    assert restored.x.meta.csdf_origin_offset_hz == 400e6
    with pytest.raises(FileExistsError):
        scp.nmr.write_csdf(dataset, target)
    scp.nmr.write_csdf(dataset, target, overwrite=True)


def test_csdf_export_rejects_irregular_grid(tmp_path):
    dataset = scp.NDDataset(
        [1.0, 2.0, 3.0], coordset=[scp.Coord([0, 1, 3], units="Hz")]
    )
    with pytest.raises(ValueError, match="linear frequency"):
        scp.nmr.write_csdf(dataset, tmp_path / "irregular.csdf")


def test_csdf_export_after_squeeze(tmp_path, spectrum):
    model, expected = spectrum
    model["dimensions"][1]["count"] = 1
    model["dependent_variables"][0]["components"] = [
        base64.b64encode(expected[:1].tobytes()).decode("ascii")
    ]
    model["dependent_variables"][0]["unit"] = "V*m"
    source = tmp_path / "source.csdf"
    source.write_text(json.dumps({"csdm": model}))
    dataset = scp.nmr.read_csdf(source).squeeze()
    target = scp.nmr.write_csdf(dataset, tmp_path / "export.csdf")

    exported = json.loads(target.read_text())["csdm"]
    assert exported["dependent_variables"][0]["unit"] == "m*V"
    restored = scp.nmr.read_csdf(target)
    np.testing.assert_array_equal(restored.data, expected[0])
    assert restored.x.meta.csdf_origin_offset_hz == 400e6
    assert restored.units == dataset.units


@pytest.mark.parametrize("unit", ["absorbance", "V**0.5"])
def test_csdf_export_rejects_nonportable_signal_units(tmp_path, unit):
    coord = scp.Coord([0, 1], units="Hz")
    dataset = scp.NDDataset([1.0, 2.0], units=unit, coordset=[coord])
    with pytest.raises(ValueError, match="portable SI signal units"):
        scp.nmr.write_csdf(dataset, tmp_path / "absorbance.csdf")


@pytest.mark.parametrize("unit", ["Bq", "rad/s"])
def test_csdf_export_rejects_non_hertz_coordinates(tmp_path, unit):
    coord = scp.Coord([0, 1], units=unit)
    dataset = scp.NDDataset([1.0, 2.0], coordset=[coord])
    with pytest.raises(ValueError, match="frequency coordinate"):
        scp.nmr.write_csdf(dataset, tmp_path / "invalid-axis.csdf")


def test_csdf_import_classifies_real_spectrum_as_processed(tmp_path):
    dataset = scp.NDDataset([1.0, 2.0], coordset=[scp.Coord([0, 1], units="Hz")])
    path = scp.nmr.write_csdf(dataset, tmp_path / "real.csdf")
    experiment = scp.nmr.Experiment(scp.nmr.read_csdf(path))
    assert experiment.source_kind == "processed_1d"
    np.testing.assert_array_equal(experiment.process().data, dataset.data)


def test_csdf_roundtrips_resistance_unit(tmp_path):
    coord = scp.Coord([0, 1], units="Hz")
    dataset = scp.NDDataset([1.0, 2.0], units="milliohm", coordset=[coord])
    path = scp.nmr.write_csdf(dataset, tmp_path / "resistance.csdf")
    exported = json.loads(path.read_text())["csdm"]
    assert exported["dependent_variables"][0]["unit"] == "mOhm"
    assert scp.nmr.read_csdf(path).units == dataset.units


def test_csdf_public_reader_retains_fine_frequency_grid(tmp_path, spectrum):
    model, _ = spectrum
    model["dimensions"][0]["count"] = 5
    model["dimensions"][0]["coordinates_offset"] = "400000000 Hz"
    model["dimensions"][0]["increment"] = "0.0000007Hz"
    values = np.arange(10, dtype="<c8").reshape(2, 5)
    model["dependent_variables"][0]["components"] = [
        base64.b64encode(values.tobytes()).decode("ascii")
    ]
    source = tmp_path / "fine-grid.csdf"
    source.write_text(json.dumps({"csdm": model}))

    spectrum = scp.nmr.read_csdf(source)
    frequencies = spectrum.x.data
    np.testing.assert_array_equal(frequencies, 400000000 + 0.0000007 * np.arange(5))
    path = scp.nmr.write_csdf(spectrum, tmp_path / "export.csdf")
    np.testing.assert_array_max_ulp(
        scp.nmr.read_csdf(path).x.data, frequencies, maxulp=2
    )
