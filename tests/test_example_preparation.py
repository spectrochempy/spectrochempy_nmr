"""Specialized reads use the importer download path only for absent examples."""

from importlib import import_module
from pathlib import Path
import shutil
from unittest.mock import Mock

import numpy as np
import pytest
import spectrochempy as scp
from spectrochempy.application.application import app
from spectrochempy.utils.exceptions import SpectroChemPyError

from test_multiplex_spam import make_spam_ser
from test_stmas import states_path as _states_path


def _files(directory):
    return {
        p.relative_to(directory): p.read_bytes()
        for p in directory.rglob("*")
        if p.is_file()
    }


@pytest.fixture(params=["states", "paired"])
def example(request, tmp_path, monkeypatch):
    source = tmp_path / "archive"
    source.mkdir()
    if request.param == "states":
        _states_path.__wrapped__(source)
        options = {"use_tdeff": True, "indirect_dwell": "IN[0]"}
    else:
        make_spam_ser(source)
        options = {"phase_cycle": "paired", "allow_partial": True}
    cache = tmp_path / "cache"
    _ = scp.preferences.datadir
    monkeypatch.setitem(app.general_preferences._trait_values, "datadir", cache)
    target = cache / "nmrdata" / "example" / "15" / "ser"
    importer = import_module("spectrochempy.core.readers.importer")

    def fetch(relative, destination, replace=False):
        assert Path(relative) == target.parent.relative_to(cache)
        assert Path(destination) == target.parent
        assert not replace
        shutil.copytree(source, destination)

    download = Mock(side_effect=fetch)
    monkeypatch.setattr(importer, "_download_from_github", download)
    return source, target, options, download


def test_specialized_read_downloads_complete_example_once(example, recwarn):
    source, target, options, download = example
    expected = scp.nmr.read(source / "ser", **options)
    first = scp.nmr.read(target, **options)
    download.assert_called_once()
    cached = _files(target.parent)
    second = scp.nmr.read(target, **options)
    download.assert_called_once()
    for result in (first, second):
        np.testing.assert_array_equal(result.data, expected.data)
        np.testing.assert_array_equal(result.x.data, expected.x.data)
        np.testing.assert_array_equal(result.y.data, expected.y.data)
        assert result.meta.acquisition == expected.meta.acquisition
        assert (Path(scp.preferences.datadir) / result.filename).resolve() == target
    assert cached == _files(target.parent)
    assert not any("dimension" in str(w.message).lower() for w in recwarn)


@pytest.mark.parametrize("missing", ["ser", "pdata/1/procs"])
def test_incomplete_local_acquisition_is_not_downloaded(example, missing):
    source, target, options, download = example
    shutil.copytree(source, target.parent)
    (target.parent / missing).unlink()
    before = _files(target.parent)
    with pytest.raises(SpectroChemPyError, match="complete experiment folder"):
        scp.nmr.read(target, **options)
    download.assert_not_called()
    assert before == _files(target.parent)


def test_absent_example_respects_local_only(example):
    _, target, options, download = example
    with pytest.raises(FileNotFoundError):
        scp.nmr.read(target, local_only=True, **options)
    download.assert_not_called()


def test_unavailable_example_reports_download_failure(example):
    _, target, options, download = example
    download.side_effect = FileNotFoundError("Example not available")
    with pytest.raises(FileNotFoundError):
        scp.nmr.read(target, **options)
    download.assert_called_once()


def test_invalid_processing_options_do_not_download(example):
    _, target, options, download = example
    with pytest.raises((ValueError, SpectroChemPyError), match="uncorrected"):
        scp.nmr.read(target, remove_digital_filter=True, **options)
    download.assert_not_called()
