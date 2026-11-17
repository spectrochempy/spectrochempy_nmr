"""Compare public JEOL COSY magnitude processing to real processed exports."""

from hashlib import sha256
import os
from pathlib import Path

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import process_jeol_cosy_magnitude
from spectrochempy_nmr import read


@pytest.fixture
def cenapt():
    directory = os.environ.get("NMR_JEOL_COSY_DIR")
    if not directory:
        pytest.skip("Set NMR_JEOL_COSY_DIR to the CENAPT collection root")
    return Path(directory)


@pytest.mark.parametrize(
    "sample, raw_hash, reference_hash, f1_size, shape, peak, tolerance, correlation",
    [
        (
            "Luteolin",
            "2cceb9c5e19ef6569c113cd6b24ca913b628ccd8dc6eb582b69dee084b6d7dfa",
            "37683c4ca03ba3a10e4f0a6cfc6103dd0cad859f4ab0381d3903e62dd9a44b09",
            None,
            (512, 6554),
            (2924.11, 2687.70),
            (9.0, 5.0),
            0.92,
        ),
        (
            "EpiCatechin",
            "f109351f2102435faa7d302659e073b77ecc1184f1c3ef8817087ca91b9df3aa",
            "b77cd1fe72cea1f6d820e40279d91cfdff3424333ca4d669e89b129c566c1152",
            2048,
            (2048, 1640),
            (2269.92, 2338.07),
            (6.0, 6.0),
            0.92,
        ),
        (
            "Rutin",
            "b3d2bd73bffe229a73c00d571e4b7355356e125410e7fe93e6af6679e6e90483",
            "41301beb97e5d23437b2c765ec3da7c9d7cbc3d8f32b386ec61ab9c3f789fc95",
            None,
            (512, 6554),
            (2459.36, 2535.91),
            (9.0, 5.0),
            0.92,
        ),
    ],
)
def test_cosy_against_processed_reference(
    cenapt,
    sample,
    raw_hash,
    reference_hash,
    f1_size,
    shape,
    peak,
    tolerance,
    correlation,
):
    (raw,) = cenapt.glob(f"*/{sample}*COSY*400MHz*.jdf")
    (reference_path,) = cenapt.glob(f"*/{sample}*COSY*400MHz*.jdx")
    assert sha256(raw.read_bytes()).hexdigest() == raw_hash
    assert sha256(reference_path.read_bytes()).hexdigest() == reference_hash

    reference = read(reference_path, protocol="interop", profile="jcampdx-spectrum-2d")
    result = process_jeol_cosy_magnitude(
        raw, f1_size=f1_size, frequency_grid=(reference.y.data, reference.x.data)
    )

    assert result.shape == shape
    assert result.meta.physical_f1_mode == "indeterminate"
    assert result.meta.isfreq == [True, True]
    assert (result.y.units, result.x.units) == (scp.ur.Hz, scp.ur.Hz)
    assert np.isrealobj(result.data) and np.all(result.data >= 0)

    np.testing.assert_array_equal(result.y.data, reference.y.data)
    np.testing.assert_array_equal(result.x.data, reference.x.data)
    actual = result.data
    assert np.corrcoef(actual.ravel(), reference.data.ravel())[0, 1] > correlation

    expected_f1, expected_f2 = peak
    i = np.flatnonzero(abs(reference.y.data - expected_f1) < 17.2)
    j = np.flatnonzero(abs(reference.x.data - expected_f2) < 5.4)
    region = actual[np.ix_(i, j)]
    a, b = np.unravel_index(region.argmax(), region.shape)
    assert abs(reference.y.data[i[a]] - expected_f1) <= tolerance[0]
    assert abs(reference.x.data[j[b]] - expected_f2) <= tolerance[1]


def test_jeol_cosy_rejects_hsqc(cenapt):
    (path,) = cenapt.glob("*/EpiCatechin*HSQC*400MHz*.jdf")
    with pytest.raises(ValueError, match="COSY JDF"):
        process_jeol_cosy_magnitude(path)


def test_jeol_cosy_default_grid(cenapt):
    (path,) = cenapt.glob("*/Luteolin*COSY*400MHz*.jdf")
    result = process_jeol_cosy_magnitude(path)
    assert result.shape == (512, 8192)
    assert np.all(np.diff(result.y.data) < 0)
    assert np.all(np.diff(result.x.data) < 0)
    assert result.y.data[255] == pytest.approx(1998.91099189125)
    assert result.x.data[4096] == pytest.approx(1998.3737105862325)

    on_native_grid = process_jeol_cosy_magnitude(
        path, frequency_grid=(result.y.data, result.x.data)
    )
    np.testing.assert_array_equal(on_native_grid.data, result.data)
