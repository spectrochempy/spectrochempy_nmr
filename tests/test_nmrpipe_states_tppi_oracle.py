"""Component-level States-TPPI regression against NMRPipe checkpoints."""

from __future__ import annotations

import hashlib
import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
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


@pytest.mark.parametrize("profile", ["states-tppi", "states"])
def test_experiment_imported_states_matches_raw_nmrpipe_checkpoints(
    monkeypatch, profile
):
    from types import SimpleNamespace

    from spectrochempy_nmr import from_interop
    from spectrochempy_nmr.experiment import Experiment
    from spectrochempy_nmr.processing import channel_interop

    fixture = ORACLE / "states_tppi_oracle.npz"
    manifest = json.loads((ORACLE / "states_tppi_oracle_manifest.json").read_text())
    assert (
        hashlib.sha256(fixture.read_bytes()).hexdigest() == manifest["fixture_sha256"]
    )
    with np.load(fixture) as oracle:
        raw, f2, tp, final = (oracle[key] for key in ("input", "f2", "tp", "final"))
    # The checkpoints zero-fill both axes; supply the same padded input.
    rows, cols = final.shape[-1], f2.shape[-1]
    padded = np.zeros((4, rows, cols), dtype=np.float32)
    for index, plane in enumerate(
        (raw[::2].real, raw[::2].imag, raw[1::2].real, raw[1::2].imag)
    ):
        padded[index, : plane.shape[0], : plane.shape[1]] = plane
    if profile == "states":
        # Undo the acquisition alternation to simulate States with the same peaks.
        padded[:, 1::2] *= -1
        f2[2::4] *= -1
        f2[3::4] *= -1
        tp[:, 1::2] *= -1
    widths = (3200.0, 10000.0)
    axes = tuple(
        SimpleNamespace(
            values=np.arange(size) / width,
            unit="s",
            label=dim,
            domain="time",
            direction="increasing",
            reference=None,
        )
        for size, width, dim in zip((rows, cols), widths, ("F1", "F2"), strict=True)
    )
    normalized = SimpleNamespace(
        data=padded,
        axes=axes,
        dimensions=("channels", "F1", "F2"),
        physical_dimensions=("F1", "F2"),
        domains=("time", "time"),
        encoding=("complex", "complex"),
        quadrature_mode="stored-components-2d",
        branch_order=("rr", "ri", "ir", "ii"),
        channel_axis=0,
        normalization_state="storage-normalized",
        normalization_profile="oracle",
        applied_operations=(),
        provenance={},
        acquisition={"spectral_width_hz": widths},
        metadata={},
        diagnostics=(),
    )
    core_fft = import_module("spectrochempy.processing.fft.fft")
    if profile == "states" and getattr(core_fft, "FFT_PREVALIDATION_API", 0) < 1:
        with pytest.raises(NotImplementedError, match="FFT prevalidation"):
            from_interop(normalized, profile=profile)
        return
    source = from_interop(normalized, profile=profile)
    stages = []
    if profile == "states":
        for dim in ("x", "y"):
            with pytest.raises(NotImplementedError, match="UNVALIDATED:STATES"):
                source.fft(dim=dim)
        normalized.acquisition["tppi_alternation"] = "declared_not_applied"
        with pytest.raises(ValueError, match="declared TPPI alternation"):
            from_interop(normalized, profile="states")
    cross_pairs = channel_interop._cross_pairs

    def record(values):
        cross = cross_pairs(values)
        stages.extend((values.copy(), cross.copy()))
        return cross

    monkeypatch.setattr(channel_interop, "_cross_pairs", record)
    spectrum = Experiment(source).process()
    for actual, expected in zip(stages[0], (f2[::2], f2[1::2]), strict=True):
        np.testing.assert_allclose(
            actual[: raw.shape[0] // 2], expected, rtol=5e-6, atol=2e-4
        )
    for actual, expected in zip(stages[1], (tp[::2], tp[1::2]), strict=True):
        np.testing.assert_allclose(
            actual[: raw.shape[0] // 2].T, expected, rtol=5e-6, atol=2e-4
        )
    rr, ri, ir, ii = spectrum.data
    np.testing.assert_allclose((rr + 1j * ir).T, final[::2], rtol=5e-6, atol=2e-4)
    np.testing.assert_allclose((ri + 1j * ii).T, final[1::2], rtol=5e-6, atol=2e-4)
    assert tuple(spectrum.c.labels) == ("rr", "ri", "ir", "ii")
    assert Experiment(source).encoding[0] == profile.upper()
    if profile == "states-tppi":
        assert spectrum.meta.tppi_alternation_state == "applied"
        assert source.meta.tppi_alternation_state == "encoded"
    else:
        assert spectrum.meta.get("tppi_alternation_state") is None
    assert Experiment(spectrum).source_kind == "processed_2d"
    with pytest.raises(NotImplementedError):
        Experiment(spectrum).process()
    with pytest.raises(ValueError, match="FFT only"):
        Experiment(source).process(phase="manual")
    for coord, width in zip((spectrum.y, spectrum.x), widths, strict=True):
        assert coord.units == scp.ur.Hz
        assert coord.data[coord.size // 2] == 0
        np.testing.assert_allclose(np.diff(coord.data), -width / coord.size, rtol=1e-12)

    np.testing.assert_array_equal(source.data.real, padded[::2])
    np.testing.assert_array_equal(source.data.imag, padded[1::2])
