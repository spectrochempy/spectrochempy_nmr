"""Public JCAMP States-family route against signed NMRPipe checkpoints."""

import sys
from pathlib import Path

import numpy as np
import pytest
from nmrglue import interop

from spectrochempy_nmr import Experiment, read


pytestmark = pytest.mark.skipif(
    '"states-tppi"'
    not in (getattr(interop, "normalize_jcampdx_fid_2d", lambda: None).__doc__ or ""),
    reason="requires development JCAMP States-family normalization",
)


def _write(path, traces, scheme):
    rows, columns = traces.shape
    text = (
        "##TITLE=Independent States reference\n##DATA TYPE=nD NMR FID\n"
        "##DATA CLASS=NTUPLES\n##NUM DIM=2\n"
        f"##.ACQUISITION SCHEME={scheme}\n##.ACQUISITION MODE=SIMULTANEOUS (DQD)\n"
        "##NTUPLES=nD NMR FID\n##SYMBOL=T1,T2,R,I\n"
        "##VAR_TYPE=INDEPENDENT,INDEPENDENT,DEPENDENT,DEPENDENT\n"
        f"##VAR_DIM={rows},{columns},{columns},{columns}\n"
        "##UNITS=SECONDS,SECONDS,ARBITRARY UNITS,ARBITRARY UNITS\n"
        "##FACTOR=0.0007,0.0001,1,1\n##FIRST=0,0,0,0\n"
        f"##LAST={(rows-1)*.0007},{(columns-1)/10000},0,0\n"
    )
    for index, trace in enumerate(traces):
        for symbol, values in [("R", trace.real), ("I", trace.imag)]:
            text += f"##PAGE=T1={index*.0007}\n##DATA TABLE=(T2++({symbol}..{symbol})), PROFILE\n"
            text += "0 " + " ".join(format(value, ".17g") for value in values) + "\n"
    path.write_text(text + "##END NTUPLES=nD NMR FID\n##END=\n")


@pytest.mark.parametrize("quadrature", ["states", "states-tppi"])
def test_states_read_reload_and_process_without_provider(
    tmp_path, monkeypatch, quadrature
):
    with np.load(
        Path(__file__).parent / "data/nmrpipe/states_tppi_oracle.npz"
    ) as oracle:
        raw, f2, final = (oracle[key] for key in ["input", "f2", "final"])
        traces = np.zeros((2 * final.shape[-1], f2.shape[-1]), dtype=complex)
        traces[: raw.shape[0], : raw.shape[1]] = raw
        if quadrature == "states":
            traces[2::4] *= -1
            traces[3::4] *= -1
        path = tmp_path / "states.dx"
        _write(path, traces, quadrature)
        source = read(
            path,
            protocol="interop",
            profile="jcampdx-fid-2d",
            conversion_profile=quadrature,
            normalize_options={
                "quadrature": quadrature,
                "conjugation": "none",
                "confirm_adjacent_pairs": True,
                "indirect_dwell_s": 1 / 3200,
            },
        )
        expected = np.stack((traces[::2], traces[1::2]))
        np.testing.assert_array_equal(source.data, expected)
        assert Experiment(source).encoding[0] == quadrature.upper()
        saved = tmp_path / "states.scp"
        source.write(saved, confirm=False)
        restored = read(saved)
        restored.meta.source_metadata = {}
        restored.origin = "changed description"
        restored.meta.normalization_profile = "changed description"
        monkeypatch.setitem(sys.modules, "nmrglue", None)
        result = Experiment(restored).process()
        reference = np.stack(
            (
                final[::2].T.real,
                final[1::2].T.real,
                final[::2].T.imag,
                final[1::2].T.imag,
            )
        )
        for observed, plane in zip(result.data, reference, strict=True):
            np.testing.assert_allclose(
                observed, plane, rtol=0, atol=5e-6 * np.max(np.abs(plane))
            )
        for coord, width in [(result.y, 3200), (result.x, 10000)]:
            np.testing.assert_allclose(
                coord.data,
                (coord.size // 2 - np.arange(coord.size)) * width / coord.size,
                rtol=0,
                atol=1e-10,
            )
        assert tuple(result.c.labels) == ("rr", "ri", "ir", "ii")
        np.testing.assert_array_equal(source.data, expected)


@pytest.mark.parametrize("quadrature", ["states", "states-tppi"])
def test_public_read_rejects_contradictory_conversion_profile(tmp_path, quadrature):
    path = tmp_path / "mismatched.dx"
    _write(path, np.ones((4, 3), dtype=complex), quadrature)
    other = "states-tppi" if quadrature == "states" else "states"
    with pytest.raises(ValueError):
        read(
            path,
            protocol="interop",
            profile="jcampdx-fid-2d",
            conversion_profile=other,
            normalize_options={
                "quadrature": quadrature,
                "conjugation": "none",
                "confirm_adjacent_pairs": True,
                "indirect_dwell_s": 0.001,
            },
        )
