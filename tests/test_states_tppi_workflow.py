"""Public read/Experiment workflow for paired Agilent acquisitions."""

import importlib

from nmrglue import interop, varian
import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr.processing.fft_validation import _require_fft_prevalidation


@pytest.fixture
def acquisition(tmp_path, request):
    try:
        _require_fft_prevalidation()
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    sequence = getattr(request, "param", "NOESY")
    directory = tmp_path / "acquisition.fid"
    directory.mkdir()
    params = {
        "array": ["phase"],
        "phase": ["1", "2"],
        "ni": ["4"],
        "arraydim": ["8"],
        "np": ["24"],
        "sw": ["8000"],
        "sw1": ["2000"],
        "sw2": ["2000"],
        "sfrq": ["500"],
        "dfrq2": ["125"],
        "ns": ["3"],
        "ne": ["2"],
        "seqfil": [sequence],
        "d2": ["0"],
    }
    procpar = {
        key: varian.create_pdic_param(key, values) for key, values in params.items()
    }
    for key in ("array", "seqfil"):
        procpar[key]["basictype"] = "2"
    varian.write_procpar(directory / "procpar", procpar)
    words = np.arange(8 * 24, dtype=">i4").reshape(8, 24)
    words[:, 1::2] *= -3
    with (directory / "fid").open("wb") as stream:
        varian.put_fileheader(stream, (8, 1, 24, 4, 96, 124, 0, 21, 1))
        for index, trace in enumerate(words, 1):
            varian.put_blockheader(stream, (0, 21, index, 0, 3, 0, 0, 0, 0))
            stream.write(trace.tobytes())
    return directory, (words[:, ::2] + 1j * words[:, 1::2]) / 6


def _positive_dft(data, axis):
    size = data.shape[axis]
    phase = np.exp(2j * np.pi * np.outer(np.arange(size), np.arange(size)) / size)
    result = np.moveaxis(np.moveaxis(data, axis, -1) @ phase.T, -1, axis)
    return np.roll(result, size // 2, axis=axis)


def _reference(samples, *, alternate):
    branches = np.stack((samples[::2].conj(), -samples[1::2].conj()))
    f2 = _positive_dft(branches, -1)
    cross = np.stack((f2[0].real + 1j * f2[1].real, f2[0].imag + 1j * f2[1].imag))
    if alternate:
        cross[:, 1::2] *= -1
    f1 = _positive_dft(cross, -2)
    return np.stack((f1[0].real, f1[1].real, f1[0].imag, f1[1].imag))


@pytest.mark.parametrize(
    "acquisition", ["NOESY", "COSY", "unrecognized"], indirect=True
)
@pytest.mark.parametrize("mode", ["states", "states-tppi"])
@pytest.mark.parametrize("target", ["directory", "fid"])
def test_read_then_experiment_uses_physical_quadrature(acquisition, mode, target):
    directory, samples = acquisition
    path = directory if target == "directory" else directory / "fid"
    fid = scp.nmr.read(path, quadrature=mode)
    before = fid.copy()
    result = scp.nmr.Experiment(fid).process()
    np.testing.assert_allclose(
        result.data,
        _reference(samples, alternate=mode == "states-tppi"),
        atol=1e-9,
        rtol=1e-12,
    )
    assert tuple(result.c.labels) == ("rr", "ri", "ir", "ii")
    assert result.x.units == result.y.units == scp.ur.Hz
    np.testing.assert_allclose(result.y.data, (2 - np.arange(4)) * 500)
    np.testing.assert_allclose(result.x.data, (6 - np.arange(12)) * (8000 / 12))
    np.testing.assert_array_equal(fid.data, before.data)
    assert fid.meta == before.meta


def test_saved_acquisition_processes_without_reader_and_extracts_components(
    acquisition, monkeypatch, tmp_path
):
    directory, samples = acquisition
    fid = scp.nmr.read(directory, quadrature="states-tppi")
    fid.write(tmp_path / "raw.scp", confirm=False)
    restored = scp.read(tmp_path / "raw.scp")

    def unavailable(*args, **kwargs):
        pytest.fail("Processing must not consult a reader or normalize source files")

    monkeypatch.setattr(interop, "read_agilent_fid_storage", unavailable)
    monkeypatch.setattr(interop, "normalize_agilent_states_2d", unavailable)
    reader = importlib.import_module("spectrochempy_nmr.readers.read_agilent")
    monkeypatch.setattr(reader, "read_varian", unavailable)
    restored.meta.source_metadata = {}
    restored.meta.seqfil = "different-experiment"
    result = scp.nmr.Experiment(restored).process()
    expected = _reference(samples, alternate=True)
    np.testing.assert_allclose(result.data, expected, atol=1e-9, rtol=1e-12)
    experiment = scp.nmr.Experiment(result)
    components = (experiment.RR, experiment.IR, experiment.RI, experiment.II)
    for index, component in enumerate(components):
        np.testing.assert_array_equal(component.data, result.data[index])
        assert component.x.units == component.y.units == scp.ur.Hz
        np.testing.assert_allclose(component.x.data, result.x.data, rtol=0, atol=1e-10)
        np.testing.assert_allclose(component.y.data, result.y.data, rtol=0, atol=1e-10)
    mask = np.zeros(result.shape, dtype=bool)
    mask[0, 1, 2] = True
    result.mask = mask
    assert scp.nmr.Experiment(result).RR.mask[1, 2]
    result.write(tmp_path / "spectrum.scp", confirm=False)
    spectrum = scp.read(tmp_path / "spectrum.scp")
    np.testing.assert_array_equal(spectrum.mask, mask)
    np.testing.assert_array_equal(spectrum.data, result.data)


def test_sequence_name_does_not_establish_quadrature(acquisition):
    directory, samples = acquisition
    fid = scp.nmr.read(directory)
    assert fid.meta.quadrature_mode == "states-stored-2d"
    with pytest.raises(NotImplementedError, match='quadrature="states-tppi"'):
        scp.nmr.Experiment(fid).process()


def test_common_processing_refuses_masked_time_samples(acquisition):
    directory, samples = acquisition
    fid = scp.nmr.read(directory, quadrature="states-tppi")
    mask = np.zeros(fid.shape, dtype=bool)
    mask[0, 1, 2] = True
    fid.mask = mask
    with pytest.raises(ValueError, match="unmasked"):
        scp.nmr.Experiment(fid).process()
