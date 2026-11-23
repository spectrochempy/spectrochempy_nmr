"""Frequency-domain JCAMP import retains the declared real plane and axes."""


import inspect

import numpy as np
import pytest
from nmrglue import interop

from spectrochempy_nmr import Experiment, read


pytestmark = pytest.mark.skipif(
    "Per-page VAR_DIM"
    not in (getattr(interop, "read_jcampdx_2d", lambda: None).__doc__ or ""),
    reason="requires development per-page JCAMP processed spectra",
)


def test_processed_topspin_layout_read_and_reload(tmp_path):
    path = tmp_path / "processed.dx"
    text = (
        "##TITLE=Processed real plane\n##DATA TYPE=nD NMR SPECTRUM\n"
        "##DATA CLASS=NTUPLES\n##NUM DIM=2\n##NTUPLES=nD NMR SPECTRUM\n"
        "##SYMBOL=F1,F2,Y\n##VAR_DIM=2,3,3\n"
        "##UNITS=HZ,HZ,ARBITRARY UNITS\n##FACTOR=5,2,-2\n"
        "##$NC_proc=5\n##FIRST=10,4,0\n##LAST=5,0,0\n"
        "##PAGE=F1=10\n##DATA TABLE=(F2++(Y..Y)),PROFILE\n2 1 2 3\n"
        "##PAGE=F1=5\n##DATA TABLE=(F2++(Y..Y)),PROFILE\n2 4 5 6\n"
        "##END NTUPLES=nD NMR SPECTRUM\n##END=\n"
    )
    path.write_text(text)
    result = read(path, protocol="interop", profile="jcampdx-spectrum-2d")
    expected = -2 * np.arange(1, 7).reshape(2, 3)
    np.testing.assert_array_equal(result.data, expected)
    np.testing.assert_array_equal(result.y.data, [10, 5])
    np.testing.assert_array_equal(result.x.data, [4, 2, 0])
    assert Experiment(result).domains == ("frequency", "frequency")
    saved = tmp_path / "spectrum.scp"
    result.write(saved, confirm=False)
    restored = read(saved)
    np.testing.assert_array_equal(restored.data, expected)
    assert Experiment(restored).domains == ("frequency", "frequency")
    assert restored.ndim == 2 and not np.iscomplexobj(restored.data)


@pytest.mark.skipif(
    "axis_reference" not in inspect.signature(interop.normalize_jcampdx_2d).parameters,
    reason="requires explicit JCAMP 2D ppm calibration",
)
@pytest.mark.parametrize("reference_source", ["processing", "acquisition", "unknown"])
def test_processed_ppm_public_read_and_scp_roundtrip(tmp_path, reference_source):
    test_processed_topspin_layout_read_and_reload(tmp_path)
    references = {
        "F1": {
            "reference_frequency_mhz": 100,
            "point_index": 1,
            "ppm": 8,
            "source": reference_source,
        },
        "F2": {
            "reference_frequency_mhz": 400,
            "point_index": 2,
            "ppm": -1,
            "source": reference_source,
        },
    }
    path = tmp_path / "processed.dx"
    path.write_text(
        path.read_text().replace("##NTUPLES=", "##.OBSERVE FREQUENCY=399\n##NTUPLES=")
    )
    result = read(
        path,
        protocol="interop",
        profile="jcampdx-spectrum-2d",
        normalize_options={"axis_reference": references},
    )
    np.testing.assert_allclose(result.y.data, [8.05, 8], rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.x.data, [-0.99, -0.995, -1], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(result.data, -2 * np.arange(1, 7).reshape(2, 3))
    assert str(result.x.units) == "ppm" and str(result.y.units) == "ppm"
    assert result.x.meta["reference_frequency"].to("MHz").magnitude == 400
    assert result.y.meta["reference_frequency"].to("MHz").magnitude == 100
    assert result.x.meta["reference_source"] == reference_source
    if reference_source == "acquisition":
        assert result.x.meta["acquisition_frequency"].to("MHz").magnitude == 400
    else:
        assert "acquisition_frequency" not in result.x.meta
        assert "acquisition_frequency" not in result.y.meta
    saved = tmp_path / "referenced.scp"
    result.write(saved, confirm=False)
    restored = read(saved)
    np.testing.assert_array_equal(restored.x.data, result.x.data)
    np.testing.assert_array_equal(restored.y.data, result.y.data)
    np.testing.assert_array_equal(restored.data, result.data)
    assert str(restored.x.units) == "ppm" and str(restored.y.units) == "ppm"
    assert Experiment(restored).domains == ("frequency", "frequency")
    assert restored.x.meta["reference_frequency"].to("MHz").magnitude == 400
    assert restored.x.meta["reference_source"] == reference_source
    if reference_source != "acquisition":
        assert "acquisition_frequency" not in restored.x.meta
