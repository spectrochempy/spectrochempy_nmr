"""Public TopSpin SER parity with the legacy decoder."""

from importlib import import_module
import os
from pathlib import Path

import nmrglue as ng
import numpy as np
import pytest
import spectrochempy as scp
from spectrochempy.utils.jsonutils import json_encoder


def _check_parity(path, monkeypatch, *, migrate, filtered, dc):
    reader = import_module("spectrochempy_nmr.readers.read_topspin")
    vendored = reader.read_fid
    calls = []
    binary_calls = []
    read_binary = ng.bruker.read

    def count_binary(directory, **kwargs):
        binary_calls.append(directory)
        return read_binary(directory, **kwargs)

    def legacy(directory, **kwargs):
        calls.append(directory)
        return vendored(directory, **kwargs)

    def old_provider(directory, *, read_pulseprogram, **kwargs):
        return vendored(directory, read_pulseprogram=read_pulseprogram, **kwargs)

    options = dict(remove_digital_filter=filtered, remove_dc_offset=dc)
    with monkeypatch.context() as previous:
        previous.setattr(
            reader, "_read_normalized_qf_ser", lambda *args, **kwargs: None
        )
        previous.setattr(ng.bruker, "read", old_provider)
        expected = scp.read_topspin(path / "ser", **options)
    with monkeypatch.context() as current:
        if path.parent.name == "exam2d_HH":
            current.setattr(
                reader, "_read_normalized_qf_ser", lambda *args, **kwargs: None
            )
        current.setattr(reader, "read_fid", legacy)
        current.setattr(ng.bruker, "read", count_binary)
        actual = scp.read_topspin(path / "ser", **options)

    assert calls == ([] if migrate else [path])
    assert len(binary_calls) <= 1
    assert actual is not None and expected is not None
    assert actual.dims == expected.dims
    assert actual.dtype == expected.dtype
    assert actual.units == expected.units
    assert json_encoder(actual.meta) == json_encoder(expected.meta)
    np.testing.assert_array_equal(actual.data, expected.data)
    for dim in actual.dims:
        got, want = actual.coordset[dim], expected.coordset[dim]
        np.testing.assert_array_equal(got.data, want.data)
        np.testing.assert_array_equal(got.labels, want.labels)
        assert got.units == want.units


@pytest.mark.parametrize(
    ("mode", "complete", "nus"),
    [(mode, complete, False) for mode in (1, 5, 6) for complete in (True, False)]
    + [(5, True, True)],
)
def test_synthetic_public_ser_parity(tmp_path, monkeypatch, mode, complete, nus):
    path = tmp_path / "bruker" / "1"
    path.mkdir(parents=True)
    (path / "acqus").write_text(
        "##TITLE= Synthetic Bruker acquisition\n"
        f"##$TD= {240 if complete else 256}\n##$SW= 16\n##$SW_h= 2000\n##$SFO1= 125\n"
        "##$BF1= 125\n##$O1= 0\n##$AQ_mod= 3\n##$DECIM= 16\n"
        "##$DSPFVS= 12\n##$GRPDLY= 2.5\n##$BYTORDA= 1\n##$DTYPA= 0\n"
        "##$PARMODE= 1\n##$NUC1= <13C>\n##$NS= 4\n##$RG= 3\n"
        f"##$FnTYPE= {2 if nus else 0}\n##END=\n"
    )
    (path / "acqu2s").write_text(
        "##TITLE= Synthetic Bruker indirect acquisition\n"
        f"##$TD= {8 if complete else 6}\n##$SW= 8\n##$SFO1= 125\n"
        f"##$FnMODE= {mode}\n##$NUC1= <13C>\n##END=\n"
    )
    samples = np.arange(8 * 256, dtype=np.int32)
    (path / "ser").write_bytes((samples * (samples % 7 - 3)).astype(">i4").tobytes())
    if nus:
        # The legacy path ignores this directory, while nmrglue-ng opens it.
        (path / "nuslist").mkdir()
    _check_parity(
        path, monkeypatch, migrate=complete and not nus, filtered=True, dc=True
    )


@pytest.mark.parametrize(
    ("reference", "migrate"),
    [
        ("exam2d_HH/1", True),
        ("exam2d_HC/3", True),
        ("topspin_2d/1", False),
    ],
)
@pytest.mark.parametrize("filtered", [False, True])
def test_local_public_ser_parity(reference, migrate, filtered, monkeypatch):
    root = os.environ.get("NMR_BRUKER_SER_DATA")
    if not root:
        pytest.skip("set NMR_BRUKER_SER_DATA to the Bruker test-data directory")
    _check_parity(
        Path(root) / reference,
        monkeypatch,
        migrate=migrate,
        filtered=filtered,
        dc=False,
    )
