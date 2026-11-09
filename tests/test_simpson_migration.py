"""Public SIMPSON migration and Tecmag reader withdrawal."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy_nmr import NMRPlugin


@pytest.mark.parametrize("kind", ["FID", "SPE"])
@pytest.mark.parametrize("ndim", [1, 2])
def test_public_simpson_returns_normalized_dataset(tmp_path, monkeypatch, kind, ndim):
    bridge = import_module("spectrochempy_nmr.interop")
    reader = import_module("spectrochempy_nmr.readers.read_simpson")
    convert = bridge.from_interop
    returned = []

    def record(result):
        dataset = convert(result)
        returned.append(dataset)
        return dataset

    def unexpected_compatibility(*args):
        pytest.fail("qualified standalone SIMPSON input took the compatibility path")

    monkeypatch.setattr(bridge, "from_interop", record)
    monkeypatch.setattr(reader.simpson, "read", unexpected_compatibility)
    path = tmp_path / f"sample.{kind.lower()}"
    header = "SIMP\nNP=8\nSW=4000\n" + ("NI=3\nSW1=1500\n" if ndim == 2 else "")
    real = np.arange(8 if ndim == 1 else 24)
    data = "".join(f"{value} {2 * value + 1}\n" for value in real)
    path.write_text(f"{header}TYPE={kind}\nDATA\n{data}END\n")

    result = scp.nmr.read_simpson(path, origin="simulation", description="reference")
    assert len(returned) == 1
    assert result is returned[0]
    assert result.origin == "simulation"
    assert result.description == "reference"
    expected = (real + 1j * (2 * real + 1)).reshape(result.shape)
    np.testing.assert_array_equal(result.data, expected)
    expected_x = np.arange(8) / 4000 if kind == "FID" else np.arange(8) * 500 - 2000
    np.testing.assert_allclose(result.x.data, expected_x)
    assert result.x.units == (scp.ur.s if kind == "FID" else scp.ur.Hz)
    if ndim == 2 and kind == "FID":
        np.testing.assert_allclose(
            result.fft(dim="x").data,
            np.fft.fftshift(np.fft.fft(expected), axes=-1),
            rtol=1e-7,
        )
        with pytest.raises(NotImplementedError, match="UNVALIDATED:F1"):
            result.fft(dim="y")


def test_simpson_input_sidecar_preserves_public_metadata(tmp_path):
    input_path = tmp_path / "sample.in"
    input_path.write_text(
        "spinsys {\n nuclei 13C\n}\npar {\n proton_frequency 400e6\n}\n"
    )
    spectrum = input_path.with_suffix(".spe")
    spectrum.write_text(
        "SIMP\nNP=4\nSW=4000\nTYPE=SPE\nDATA\n1 2\n3 4\n5 6\n7 8\nEND\n"
    )

    dataset = scp.nmr.read_simpson(input_path)
    assert dataset.meta.nucleus == ["13C"]
    assert dataset.meta.sfrq == [400.0]
    assert dataset.x.size == 4
    np.testing.assert_array_equal(dataset.data, [1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j])


def test_short_indexed_simpson_retains_compatibility(tmp_path):
    path = tmp_path / "short.spe"
    path.write_text("0 1 2\n5 3 4\n")
    dataset = scp.nmr.read_simpson(path)
    assert dataset.meta.isfreq == [True]
    np.testing.assert_array_equal(dataset.data, [1 + 2j, 3 + 4j])


def test_tecmag_reader_is_not_registered():
    plugin = NMRPlugin()
    assert "tecmag" not in plugin.io_namespaces
    assert "tecmag" not in {reader["name"] for reader in plugin.register_readers()}
    package = import_module("spectrochempy_nmr")
    assert not hasattr(package, "read_tecmag")
    assert package._infer_nmr_filetype_key(Path("sample.tnt")) is None
