"""Frequency-domain JCAMP import retains the declared real plane and axes."""


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
