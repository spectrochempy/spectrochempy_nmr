"""Native reader contracts independent of external scientific data files."""

from importlib import import_module

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr.nmr_metadata import extract_topspin_metadata


def _bruker_dictionary():
    return {
        "acqus": {
            "PARMODE": 1,
            "AQ_mod": 3,
            "DECIM": 1,
            "TD": 8,
            "SW": 10.0,
            "SFO1": 400.0,
            "BF1": 400.0,
            "SF": 400.0,
            "RG": 2.0,
            "NS": 4,
            "NUC1": "1H",
        },
        "acqu2s": {
            "FnMODE": 5,
            "TD": 6,
            "SW": 10.0,
            "SFO1": 100.0,
            "BF1": 100.0,
            "SF": 100.0,
            "NUC1": "13C",
        },
    }


def _bruker_raw(tmp_path, monkeypatch):
    reader = import_module("spectrochempy_nmr.readers.read_topspin")
    experiment = tmp_path / "sample" / "1"
    experiment.mkdir(parents=True)
    path = experiment / "ser"
    path.touch()
    raw = np.arange(24).reshape(6, 4) + 1j * np.arange(24, 48).reshape(6, 4)
    monkeypatch.setattr(
        reader, "_read_raw_data", lambda *a, **k: (_bruker_dictionary(), raw.copy())
    )
    dataset = reader._read_topspin(scp.NDDataset(), path, remove_digital_filter=False)
    return dataset, raw


def test_bruker_raw_preserves_all_samples_and_physical_metadata(tmp_path, monkeypatch):
    dataset, raw = _bruker_raw(tmp_path, monkeypatch)
    assert dataset.dims == ["c", "y", "x"]
    assert dataset.shape == (2, 3, 4)
    assert dataset.c.labels.tolist() == ["branch_0", "branch_1"]
    np.testing.assert_allclose(dataset.data[0], raw[0::2].conj() / 8)
    np.testing.assert_allclose(dataset.data[1], raw[1::2].conj() / 8)
    metadata = extract_topspin_metadata(dataset.meta)
    assert metadata.ndim == 2
    assert metadata.domains == ("time", "time")
    assert metadata.encoding == ("STATES-TPPI", "DQD")
    assert metadata.nuclei == ("13C", "1H")


def test_bruker_guards_component_axis_and_transform_order(tmp_path, monkeypatch):
    dataset, _ = _bruker_raw(tmp_path, monkeypatch)
    original = dataset.data.copy()
    with pytest.raises(NotImplementedError, match="component axis"):
        dataset.fft(dim="c")
    np.testing.assert_array_equal(dataset.data, original)
    with pytest.raises(NotImplementedError, match="PENDING-F2"):
        dataset.fft(dim="y")
    f2 = dataset.fft(dim="x")
    assert f2.meta.stage == "F2-spectrum"
    assert f2.meta.encoding[1] == "INDIRECT:STATES-TPPI"
    result = f2.fft(dim="y")
    assert result.meta.stage == "spectrum"
    assert result.c.labels.tolist() == ["F1_R", "F1_I"]
    assert extract_topspin_metadata(result.meta).domains == ("frequency", "frequency")
    np.testing.assert_array_equal(dataset.data, original)
    with pytest.raises(NotImplementedError, match="Inverse"):
        f2.ifft(dim="x")


def _assert_rejected_transform_preserves_dataset(dataset, before):
    np.testing.assert_array_equal(dataset.data, before.data)
    np.testing.assert_array_equal(dataset.mask, before.mask)
    assert dataset.shape == before.shape
    assert dataset.dims == before.dims
    assert dataset.meta == before.meta
    assert dataset.meta.readonly == before.meta.readonly
    assert dataset.history == before.history
    assert dataset.units == before.units
    for dim in before.dims:
        actual, expected = dataset.coordset[dim], before.coordset[dim]
        np.testing.assert_array_equal(actual.data, expected.data)
        np.testing.assert_array_equal(actual.labels, expected.labels)
        assert actual.units == expected.units
        assert actual.title == expected.title
        assert actual.meta == expected.meta


@pytest.mark.parametrize("entry", ["method", "functional"])
@pytest.mark.parametrize("size", [None, 2, 8])
@pytest.mark.parametrize("inverse", [False, True])
def test_rejected_inplace_channel_fft_preserves_dataset(
    tmp_path, monkeypatch, entry, size, inverse
):
    dataset, _ = _bruker_raw(tmp_path, monkeypatch)
    if inverse:
        dataset = dataset.fft(dim="x")
    before = dataset.copy()
    name = "ifft" if inverse else "fft"
    kwargs = {"dim": "x" if inverse else "y", "inplace": True, "size": size}
    with pytest.raises(NotImplementedError, match="Inverse|PENDING-F2"):
        if entry == "method":
            getattr(dataset, name)(**kwargs)
        else:
            getattr(scp, name)(dataset, **kwargs)
    _assert_rejected_transform_preserves_dataset(dataset, before)


@pytest.mark.parametrize("entry", ["method", "functional"])
@pytest.mark.parametrize("size", [None, 2, 8])
@pytest.mark.parametrize(
    "encoding", ["UNVALIDATED:AGILENT-STATES", "UNVALIDATED:JEOL-STATES"]
)
def test_unvalidated_inplace_fft_preserves_dataset(
    tmp_path, monkeypatch, entry, size, encoding
):
    dataset, _ = _bruker_raw(tmp_path, monkeypatch)
    dataset.meta.readonly = False
    dataset.meta.encoding[1] = encoding
    dataset.meta.readonly = True
    before = dataset.copy()
    with pytest.raises(NotImplementedError, match="UNVALIDATED"):
        if entry == "method":
            dataset.fft(axis=1, inplace=True, size=size)
        else:
            scp.fft(dataset, axis=1, inplace=True, size=size)
    _assert_rejected_transform_preserves_dataset(dataset, before)


def test_bruker_processed_preserves_four_components_and_persistence(
    tmp_path, monkeypatch
):
    reader = import_module("spectrochempy_nmr.readers.read_topspin")
    path = tmp_path / "sample" / "1" / "pdata" / "1" / "2rr"
    path.parent.mkdir(parents=True)
    path.touch()
    components = [np.arange(12).reshape(3, 4) + 100 * n for n in range(4)]
    dic = _bruker_dictionary()
    dic.update({"procs": {"SI": 4}, "proc2s": {"SI": 3}})
    monkeypatch.setattr(
        reader, "_read_pdata_fallback", lambda *a, **k: (dic, components)
    )
    result = reader._read_topspin(scp.NDDataset(), path)
    np.testing.assert_allclose(result.data[0], (components[0] + 1j * components[2]) / 2)
    np.testing.assert_allclose(result.data[1], (components[1] + 1j * components[3]) / 2)
    assert result.meta.stage == "spectrum"
    path = result.save_as(tmp_path / "native", confirm=False)
    restored = scp.NDDataset.load(path)
    np.testing.assert_array_equal(restored.data, result.data)
    assert restored.dims == ["c", "y", "x"]
    assert restored.c.labels.tolist() == ["S_R", "S_I"]


def test_physical_axis_reordering_keeps_fourier_conventions(tmp_path, monkeypatch):
    dataset, _ = _bruker_raw(tmp_path, monkeypatch)
    expected = dataset.fft(dim="x").fft(dim="y")
    reordered = dataset.swapdims("x", "y").fft(dim="x").fft(dim="y")
    np.testing.assert_allclose(reordered.swapdims("x", "y").data, expected.data)
    assert extract_topspin_metadata(reordered.meta).encoding == ("DQD", "STATES-TPPI")


def test_agilent_storage_decoding_does_not_claim_quadrature_normalization(
    tmp_path, monkeypatch
):
    reader = import_module("spectrochempy_nmr.readers.read_agilent")
    path = tmp_path / "agilent" / "fid"
    path.parent.mkdir()
    path.touch()
    path.with_name("procpar").touch()
    raw = np.arange(24).reshape(6, 4) + 1j * np.arange(24, 48).reshape(6, 4)
    procpar = {
        key: {"values": values}
        for key, values in {
            "np": ["8"],
            "ni": ["3"],
            "arraydim": ["6"],
            "phase": ["1", "2"],
            "seqfil": ["NOESY"],
            "sw": ["4000"],
            "sw2": ["1000"],
        }.items()
    }
    monkeypatch.setattr(
        reader,
        "read_varian",
        lambda *a, **k: ({"procpar": procpar, "ntraces": 1, "np": 8}, raw.copy()),
    )
    result = reader._read_agilent(scp.NDDataset(), path)
    np.testing.assert_array_equal(result.data[0], raw[0::2])
    np.testing.assert_array_equal(result.data[1], raw[1::2])
    assert result.meta.normalization_state == "storage-decoded"
    with pytest.raises(NotImplementedError, match="UNVALIDATED"):
        result.fft(dim="x").fft(dim="y")
    annotated = scp.NDDataset(
        raw,
        meta={
            "seqfil": "NOESY",
            "phase": (1, 2),
            "ni": 3,
            "np": 8,
            "arraydim": 6,
            "ntraces": 1,
        },
    )
    expected = scp.nmr.process_agilent_noesy_states_tppi(annotated)
    experimental = scp.nmr.process_agilent_noesy_states_tppi(result)
    np.testing.assert_allclose(experimental.data, expected.data)
    assert experimental.meta.processing_profile == "states-tppi-experimental"
    f2 = result.fft(dim="x")
    assert result.meta.stage == "acquisition"
    assert result.meta.isfreq == [False, False, False]
    assert f2.meta.stage == "F2-spectrum"
    assert f2.meta.isfreq == [False, False, True]
    with pytest.raises(ValueError, match="time-domain"):
        scp.nmr.process_agilent_noesy_states_tppi(f2)
    f2.meta.readonly = False
    f2.meta.stage = "acquisition"
    with pytest.raises(ValueError, match="time-domain"):
        scp.nmr.process_agilent_noesy_states_tppi(f2)


def test_jeol_retains_decoded_matrix_without_fabricating_channels(
    tmp_path, monkeypatch
):
    reader = import_module("spectrochempy_nmr.readers.read_jeol")
    path = tmp_path / "test.jdf"
    path.touch()
    raw = np.arange(24).reshape(4, 6) + 1j * np.arange(24, 48).reshape(4, 6)
    dic = {
        "header": {"data_axis_type": ["complex", "complex"]},
        "parameters": {"x_sweep": 1000.0, "y_sweep": 4000.0},
    }
    monkeypatch.setattr(reader, "_read_jeol_raw", lambda *a: (dic, raw.copy()))
    result = reader._read_jdf(scp.NDDataset(), path)
    np.testing.assert_array_equal(result.data, raw)
    assert result.meta.normalization_state == "storage-decoded"
    with pytest.raises(NotImplementedError, match="UNVALIDATED:JEOL"):
        result.fft(dim="y")


def test_jeol_real_qf_reader_preserves_fft_support(tmp_path, monkeypatch):
    reader = import_module("spectrochempy_nmr.readers.read_jeol")
    path = tmp_path / "real.jdf"
    path.touch()
    raw = np.arange(8, dtype=float)
    dic = {
        "header": {"data_axis_type": ["real"]},
        "parameters": {"x_sweep": 1000.0},
    }
    monkeypatch.setattr(reader, "_read_jeol_raw", lambda *a: (dic, raw.copy()))
    dataset = reader._read_jdf(scp.NDDataset(), path)
    assert dataset.meta.encoding == ["QF"]
    assert dataset.dtype.kind == "f"
    transformed = dataset.fft(dim="x")
    expected = np.fft.fftshift(np.fft.fft(raw))
    np.testing.assert_allclose(transformed.data, expected)
    np.testing.assert_array_equal(dataset.data, raw)
