"""Public JCAMP QF route into the existing source-independent 2D processor."""

import numpy as np
import pytest

from nmrglue import interop
from spectrochempy_nmr import Experiment, read


pytestmark = pytest.mark.skipif(
    not hasattr(interop, "read_jcampdx_fid_2d"),
    reason="requires development nmrglue-ng jcampdx-fid-2d profile",
)


def _write_qf(path):
    samples = np.array([[1 + 2j, 3 - 4j, 5 + 6j], [7 - 8j, 9 + 10j, 11 - 12j]])
    header = (
        "##TITLE=Synthetic QF\n##DATA TYPE=nD NMR FID\n##DATA CLASS=NTUPLES\n"
        "##NUM DIM=2\n##.ACQUISITION SCHEME=NOT PHASE SENSITIVE\n"
        "##NTUPLES=nD NMR FID\n##SYMBOL=T1,T2,R,I\n"
        "##VAR_TYPE=INDEPENDENT,INDEPENDENT,DEPENDENT,DEPENDENT\n"
        "##VAR_DIM=2,3,3,3\n##UNITS=SECONDS,SECONDS,ARBITRARY UNITS,ARBITRARY UNITS\n"
        "##FACTOR=0.002,0.001,1,1\n##FIRST=0,0,1,2\n##LAST=0.002,0.002,11,-12\n"
    )
    for index, row in enumerate(samples):
        for symbol, values in (("R", row.real), ("I", row.imag)):
            header += f"##PAGE=T1={index * .002}\n##DATA TABLE=(T2++({symbol}..{symbol})), PROFILE\n"
            header += "0 " + " ".join(str(value) for value in values) + "\n"
    path.write_text(header + "##END NTUPLES=nD NMR FID\n##END=\n")
    return samples


def test_public_qf_read_process_and_reload(tmp_path):
    path = tmp_path / "fid.dx"
    expected = _write_qf(path)
    raw = read(
        path,
        protocol="interop",
        profile="jcampdx-fid-2d",
        normalize_options={"quadrature": "qf", "conjugation": "none"},
    )
    np.testing.assert_array_equal(raw.data, expected)
    assert Experiment(raw).encoding == ("QF", "DQD")
    assert Experiment(raw).domains == ("time", "time")
    saved = tmp_path / "raw.scp"
    raw.write(saved, confirm=False)
    restored = read(saved)
    assert Experiment(restored).encoding == ("QF", "DQD")
    result = Experiment(restored).process(size=(4, 6))
    padded = np.pad(expected, ((0, 2), (0, 3)))
    f2 = np.exp(2j * np.pi * np.outer(np.arange(6), np.arange(6)) / 6)
    f1 = np.exp(-2j * np.pi * np.outer(np.arange(4), np.arange(4)) / 4)
    direct = np.fft.fftshift(padded @ f2, axes=1)
    reference = np.fft.fftshift((f1 @ direct).conj(), axes=0)
    np.testing.assert_allclose(result.data, reference, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result.x.data,
        [500, 1000 / 3, 1000 / 6, 0, -1000 / 6, -1000 / 3],
        rtol=1e-14,
        atol=1e-12,
    )
    np.testing.assert_array_equal(result.y.data, [250, 125, 0, -125])
    assert Experiment(result).domains == ("frequency", "frequency")
    np.testing.assert_array_equal(raw.data, expected)


def test_public_route_refuses_phase_sensitive_qf_claim(tmp_path):
    path = tmp_path / "echo.dx"
    _write_qf(path)
    path.write_text(path.read_text().replace("NOT PHASE SENSITIVE", "Echo-Antiecho"))
    with pytest.raises(ValueError, match="declared QF"):
        read(
            path,
            protocol="interop",
            profile="jcampdx-fid-2d",
            normalize_options={"quadrature": "qf", "conjugation": "none"},
        )
