"""Synthetic tests for the experimental Agilent interop facade."""

from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import read_agilent_interop


def _normalized(source, *, spectral_width=2000.0, observation_frequency=125.0, points=4):
    data = np.arange(points, dtype=np.float32) + 1j * np.arange(points, dtype=np.float32)[::-1]
    return SimpleNamespace(
        data=data.astype(np.complex64),
        axes=(
            SimpleNamespace(
                values=np.arange(points, dtype=float) / spectral_width,
                unit="s",
                label="H1",
                domain="time",
                direction="increasing",
                reference=SimpleNamespace(
                    observation_frequency=observation_frequency,
                    observation_frequency_unit="MHz",
                    offset=None,
                    offset_unit=None,
                    source="acquisition",
                ),
            ),
        ),
        dimensions=("F2",),
        physical_dimensions=("F2",),
        domains=("time",),
        quadrature_mode="complex-1d",
        branch_order=("complex",),
        normalization_state="storage-normalized",
        normalization_profile="agilent-fid-1d-complex-v1",
        applied_operations=(
            {
                "name": "decode",
                "stage": "acquisition",
                "input_dimensions": ("storage",),
                "output_dimensions": ("F2",),
                "axis": "F2",
                "scale": None,
                "conjugation": "none",
                "sign": "none",
                "permutation": None,
                "evidence": "synthetic",
            },
        ),
        provenance={"source": str(source), "format": "agilent-fid"},
        acquisition={
            "spectral_width_hz": spectral_width,
            "dwell_time_s": 1 / spectral_width,
            "observation_frequency_mhz": observation_frequency,
        },
        metadata={},
        encoding=("QSIM",),
        diagnostics=(
            SimpleNamespace(
                code="acquisition.note",
                severity="info",
                message="synthetic diagnostic",
                context={"source": str(source)},
            ),
        ),
        interop_axes=(
            {
                "label": "H1",
                "unit": "s",
                "domain": "time",
                "direction": "increasing",
                "reference": {
                    "observation_frequency": 125.0,
                    "observation_frequency_unit": "MHz",
                    "offset": None,
                    "offset_unit": None,
                    "source": "acquisition",
                },
            },
        ),
        interop_domains=("time",),
        interop_encoding=("QSIM",),
    )


def _install_fake_nmrglue(monkeypatch, normalized_by_path):
    interop = ModuleType("nmrglue.interop")

    def read_agilent_fid(path):
        return normalized_by_path[path]

    interop.read_agilent_fid = read_agilent_fid
    interop.normalize_agilent_fid = lambda raw: raw
    package = ModuleType("nmrglue")
    package.interop = interop
    monkeypatch.setitem(__import__("sys").modules, "nmrglue", package)


def _acquisition(tmp_path, name):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "fid").write_bytes(b"synthetic")
    (directory / "procpar").write_text("synthetic")
    return directory


def test_facade_reads_directory_fid_and_deduplicates(monkeypatch, tmp_path):
    directory = _acquisition(tmp_path, "one")
    normalized = _normalized(directory)
    _install_fake_nmrglue(monkeypatch, {str(directory.resolve()): normalized})

    result = read_agilent_interop(directory, directory / "fid")

    assert isinstance(result, scp.NDDataset)
    assert result.shape == (4,)
    assert result.meta.provenance["source"] == str(directory)


def test_facade_merge_false_preserves_individual_order(monkeypatch, tmp_path):
    first = _acquisition(tmp_path, "first")
    second = _acquisition(tmp_path, "second")
    normalized = {
        str(first.resolve()): _normalized(first),
        str(second.resolve()): _normalized(second),
    }
    _install_fake_nmrglue(monkeypatch, normalized)

    result = read_agilent_interop([first, second], merge=False)

    assert isinstance(result, scp.ScpObjectList)
    assert [item.meta.provenance["source"] for item in result] == [
        str(first),
        str(second),
    ]


def test_facade_merge_true_uses_core_stack_and_records_series(monkeypatch, tmp_path):
    first = _acquisition(tmp_path, "first")
    second = _acquisition(tmp_path, "second")
    normalized = {
        str(first.resolve()): _normalized(first),
        str(second.resolve()): _normalized(second),
    }
    _install_fake_nmrglue(monkeypatch, normalized)
    source_snapshots = []
    original_stack = scp.stack

    def recording_stack(*datasets):
        source_snapshots.extend(
            (dataset.data.copy(), dict(dataset.meta.provenance), dict(dataset.meta.acquisition))
            for dataset in datasets
        )
        result = original_stack(*datasets)
        for dataset, (data, provenance, acquisition) in zip(
            datasets, source_snapshots, strict=True
        ):
            np.testing.assert_array_equal(dataset.data, data)
            assert dict(dataset.meta.provenance) == provenance
            assert dict(dataset.meta.acquisition) == acquisition
        return result

    monkeypatch.setattr(scp, "stack", recording_stack)

    result = read_agilent_interop(first, second, merge=True)

    assert result.shape == (2, 4)
    assert result.meta.series_kind == "independent_acquisitions"
    assert result.meta.series_axis_role == "acquisition_index"
    assert result.meta.series_coordinate == "index"
    assert result.meta.acquisition_order == (str(first), str(second))
    assert result.meta.normalization_state == "storage-normalized"
    assert result.meta.normalization_profile == "agilent-fid-1d-complex-v1"
    assert len(result.meta.individual_normalization_history) == 2
    assert result.meta.individual_normalization_history[0]["operations"][0]["name"] == "decode"
    assert result.meta.individual_normalization_history[0]["axes"][0]["reference"]["observation_frequency"] == 125.0
    assert result.meta.individual_normalization_history[0]["diagnostics"][0]["code"] == "acquisition.note"
    assert result.meta.individual_normalization_history[0]["domains"] == ("time",)
    assert len(source_snapshots) == 2
    assert result.history[-1].endswith("Stacked 2 normalized Agilent acquisitions")


def test_facade_rejects_incompatible_merge_and_2d_profile(monkeypatch, tmp_path):
    first = _acquisition(tmp_path, "first")
    second = _acquisition(tmp_path, "second")
    normalized = {
        str(first.resolve()): _normalized(first, spectral_width=2000.0),
        str(second.resolve()): _normalized(second, spectral_width=2500.0),
    }
    _install_fake_nmrglue(monkeypatch, normalized)

    with pytest.raises(ValueError, match="spectral_width_hz"):
        read_agilent_interop(first, second, merge=True)

    normalized[str(second.resolve())] = _normalized(second, spectral_width=2000.0)
    normalized[str(second.resolve())].acquisition["carrier_hz"] = 12000.0
    normalized[str(first.resolve())].acquisition["carrier_hz"] = 11000.0
    with pytest.raises(ValueError, match="carrier_hz"):
        read_agilent_interop(first, second, merge=True)

    normalized[str(second.resolve())] = _normalized(
        second, observation_frequency=126.0
    )
    normalized[str(first.resolve())] = _normalized(
        first, observation_frequency=125.0
    )
    with pytest.raises(ValueError, match="observation_frequency_mhz"):
        read_agilent_interop(first, second, merge=True)

    normalized[str(second.resolve())] = _normalized(second)
    normalized[str(first.resolve())] = _normalized(first)
    normalized[str(second.resolve())].acquisition["carrier_hz"] = 12000.0
    normalized[str(first.resolve())].acquisition["carrier_hz"] = 11000.0
    with pytest.raises(ValueError, match="carrier_hz"):
        read_agilent_interop(first, second, merge=True)

    normalized[str(second.resolve())] = _normalized(second)
    normalized[str(first.resolve())] = _normalized(first)
    normalized[str(second.resolve())].axes[0].reference.observation_frequency = 126.0
    with pytest.raises(ValueError, match="interop_axes"):
        read_agilent_interop(first, second, merge=True)

    (tmp_path / "two_d").mkdir()
    (tmp_path / "two_d" / "fid").write_bytes(b"synthetic")
    (tmp_path / "two_d" / "procpar").write_text("synthetic")
    bad = _normalized(tmp_path / "two_d")
    bad.data = np.zeros((2, 2), dtype=np.complex64)
    normalized[str((tmp_path / "two_d").resolve())] = bad
    with pytest.raises(ValueError, match="1D"):
        read_agilent_interop(tmp_path / "two_d")
