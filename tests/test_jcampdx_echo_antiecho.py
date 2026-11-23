"""Public JCAMP echo/anti-echo processing against independent NMRPipe data."""

import sys
from inspect import signature
from pathlib import Path

import numpy as np
import pytest
from nmrglue import interop

from spectrochempy_nmr import Experiment, read


pytestmark = pytest.mark.skipif(
    "confirm_adjacent_pairs"
    not in signature(
        getattr(interop, "normalize_jcampdx_fid_2d", lambda: None)
    ).parameters,
    reason="requires development JCAMP echo/anti-echo normalization",
)


@pytest.mark.parametrize("digital_filter", [False, True])
def test_jcamp_route_matches_nmrpipe_after_reload(
    tmp_path, monkeypatch, digital_filter
):
    name = (
        "echo_antiecho_dmx_oracle.npz" if digital_filter else "echo_antiecho_oracle.npz"
    )
    with np.load(Path(__file__).parent / "data/nmrpipe" / name) as oracle:
        channels = oracle["raw"]
        traces = np.empty((2 * channels.shape[1], channels.shape[2]), dtype=complex)
        traces[::2], traces[1::2] = channels
        rows, columns = traces.shape
        text = (
            "##TITLE=Independent echo/anti-echo reference\n##DATA TYPE=nD NMR FID\n"
            "##DATA CLASS=NTUPLES\n##NUM DIM=2\n"
            "##.ACQUISITION SCHEME=Echo-Antiecho\n"
            "##.ACQUISITION MODE=SIMULTANEOUS (DQD)\n"
            f"##$GRPDLY={70 + 1 / 6}\n##NTUPLES=nD NMR FID\n"
            "##SYMBOL=T1,T2,R,I\n"
            "##VAR_TYPE=INDEPENDENT,INDEPENDENT,DEPENDENT,DEPENDENT\n"
            f"##VAR_DIM={rows},{columns},{columns},{columns}\n"
            "##UNITS=SECONDS,SECONDS,ARBITRARY UNITS,ARBITRARY UNITS\n"
            "##FACTOR=0.0007,0.00025,1,1\n##FIRST=0,0,0,0\n"
            f"##LAST={(rows - 1) * .0007},{(columns - 1) / 4000},0,0\n"
        )
        for index, trace in enumerate(traces):
            for component, values in (("R", trace.real), ("I", trace.imag)):
                text += f"##PAGE=T1={index * .0007}\n##DATA TABLE=(T2++({component}..{component})), PROFILE\n"
                text += (
                    "0 " + " ".join(format(value, ".17g") for value in values) + "\n"
                )
        path = tmp_path / "reference.dx"
        path.write_text(text + "##END NTUPLES=nD NMR FID\n##END=\n")
        raw = read(
            path,
            protocol="interop",
            profile="jcampdx-fid-2d",
            normalize_options={
                "quadrature": "echo-antiecho",
                "conjugation": "none",
                "confirm_adjacent_pairs": True,
                "indirect_dwell_s": 0.001,
            },
        )
        np.testing.assert_array_equal(raw.data, channels)
        assert tuple(raw.c.labels) == ("first", "second")
        saved = tmp_path / "raw.scp"
        raw.write(saved, confirm=False)
        restored = read(saved)
        restored.meta.source_metadata = {}
        restored.origin = "changed descriptive origin"
        restored.meta.normalization_profile = "changed descriptive profile"
        monkeypatch.setitem(sys.modules, "nmrglue", None)
        result = Experiment(restored).process(
            **({"digital_filter": "phase"} if digital_filter else {})
        )
        tolerance = 2e-5 if digital_filter else 1e-6
        for plane, expected in zip(result.data, oracle["spectrum"], strict=True):
            np.testing.assert_allclose(
                plane, expected, rtol=0, atol=tolerance * np.max(np.abs(expected))
            )
        for coord, dimension in ((result.y, "f1"), (result.x, "f2")):
            np.testing.assert_allclose(
                coord.data, oracle[f"hz_{dimension}"], rtol=0, atol=5e-4
            )
        assert tuple(result.c.labels) == ("rr", "ri", "ir", "ii")
        assert Experiment(result).domains == ("frequency", "frequency")
        np.testing.assert_array_equal(raw.data, channels)
