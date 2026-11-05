"""Component-level States-TPPI regression against NMRPipe checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import spectrochempy as scp


ORACLE = Path(__file__).parent / "data" / "nmrpipe"


def _dataset(components):
    return scp.NDDataset(
        np.stack(
            [
                components[..., 0] + 1j * components[..., 1],
                components[..., 2] + 1j * components[..., 3],
            ]
        ),
        dims=["c", "y", "x"],
        coordset=[
            scp.Coord.arange(2),
            scp.Coord.arange(components.shape[0], unit="s"),
            scp.Coord.arange(components.shape[1], unit="s"),
        ],
        meta={
            "encoding": ["CHANNELS", "STATES-TPPI", "DQD"],
            "iscomplex": [False, True, True],
            "td": [2, *components.shape[:2]],
            "si": [2, 64, 512],
            "phc0": [0, 0, 0],
            "phc1": [0, 0, 0],
            "phased": [False, False, False],
        },
    )


def test_states_tppi_nmrpipe_checkpoint_preserves_all_components():
    manifest = json.loads((ORACLE / "states_tppi_oracle_manifest.json").read_text())
    fixture = ORACLE / "states_tppi_oracle.npz"
    assert (
        hashlib.sha256(fixture.read_bytes()).hexdigest() == manifest["fixture_sha256"]
    )
    assert manifest["encoding"] == {"yMODE": "States-TPPI", "yALT": 2, "aq2D": "States"}

    with np.load(fixture) as oracle:
        for name, details in manifest["files"].items():
            assert oracle[name].shape == tuple(details["shape"])
            assert str(oracle[name].dtype) == details["dtype"]
        input_components = oracle["input_components"]
        f2_expected = oracle["f2_components"]
        tp_expected = oracle["tp_cross"]
        final_expected = oracle["final_components"]

    assert np.all(np.isfinite(input_components))
    assert all(np.linalg.norm(input_components[..., i]) > 0 for i in range(4))

    dataset = _dataset(input_components)
    f2 = dataset.fft(dim="x", size=512)
    f2_components = np.stack(
        [f2.data[0].real, f2.data[0].imag, f2.data[1].real, f2.data[1].imag], axis=-1
    )
    np.testing.assert_allclose(f2_components, f2_expected, rtol=5e-6, atol=2e-4)

    # The TP checkpoint is compared before F1 processing, without collapsing
    # the four-component data to a magnitude.
    f2_cross = np.stack(
        [
            f2_components[..., 0] + 1j * f2_components[..., 2],
            f2_components[..., 1] + 1j * f2_components[..., 3],
        ],
        axis=-1,
    )
    np.testing.assert_allclose(f2_cross, tp_expected, rtol=5e-6, atol=2e-4)

    final = f2.fft(dim="y", size=64)
    final_components = np.stack(
        [
            final.data[0].real,
            final.data[0].imag,
            final.data[1].real,
            final.data[1].imag,
        ],
        axis=-1,
    )
    np.testing.assert_allclose(final_components, final_expected, rtol=5e-6, atol=2e-4)

    # Every canonical component has an independent nonzero signal; no
    # magnitude-only comparison can make this test pass.
    for component in range(4):
        assert np.linalg.norm(final_components[..., component]) > 0
