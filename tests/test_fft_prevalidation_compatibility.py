"""Old cores keep 1D support but cannot create unsafe guarded NMR profiles."""

from importlib import import_module

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy.utils.meta import Meta
from spectrochempy_nmr.processing.complex_channels import process_complex_channels
from spectrochempy_nmr.readers._channels import set_channel_data


@pytest.fixture
def no_prevalidation(monkeypatch):
    module = import_module("spectrochempy.processing.fft.fft")
    monkeypatch.setattr(module, "FFT_PREVALIDATION_API", 0, raising=False)


@pytest.mark.usefixtures("no_prevalidation")
def test_channel_creation_rejects_old_core_without_modifying_dataset_or_metadata():
    dataset = scp.NDDataset(np.arange(8))
    before = dataset.copy()
    meta = Meta(encoding=["STATES", "DQD"], td=[3, 4])
    meta_before = meta.copy()
    with pytest.raises(NotImplementedError, match="FFT prevalidation"):
        set_channel_data(
            dataset,
            np.ones((2, 3, 4), dtype=complex),
            [scp.Coord.arange(3), scp.Coord.arange(4)],
            meta,
            labels=("branch_0", "branch_1"),
            stage="acquisition",
            profile="test",
        )
    np.testing.assert_array_equal(dataset.data, before.data)
    assert dataset.dims == before.dims
    assert dataset.meta == before.meta
    assert meta == meta_before


@pytest.mark.usefixtures("no_prevalidation")
def test_normalized_channel_processing_rejects_old_core_before_any_operation():
    dataset = scp.NDDataset(np.ones((2, 3, 4), dtype=complex), dims=["c", "y", "x"])
    before = dataset.copy()
    with pytest.raises(NotImplementedError, match="FFT prevalidation"):
        process_complex_channels(dataset, reconstruction="states-tppi")
    np.testing.assert_array_equal(dataset.data, before.data)
    assert dataset.meta == before.meta


@pytest.mark.usefixtures("no_prevalidation")
@pytest.mark.parametrize("dtype", [float, complex])
def test_old_core_keeps_ordinary_1d_qf_support(dtype):
    data = np.arange(8).astype(dtype)
    dataset = scp.NDDataset(
        data,
        coordset=[scp.Coord.arange(8, units="s")],
        meta={"encoding": ["QF"], "isfreq": [False], "td": [8]},
    )
    spectrum = dataset.fft()
    np.testing.assert_allclose(spectrum.data, np.fft.fftshift(np.fft.fft(data.conj())))
    np.testing.assert_array_equal(dataset.data, data)
