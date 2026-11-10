"""Qualify public interop routing with pinned provider fixtures."""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import interop


def reference(name, checksum):
    root = os.environ.get("NMR_INTEROP_QUALIFICATION_DATA")
    if not root:
        pytest.skip("configure the checksum-pinned interop fixtures")
    path = Path(root) / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum
    return path


def test_public_jcampdx_1d_returns_converted_dataset(monkeypatch):
    from nmrglue import jcampdx

    path = reference(
        "epicatechin_1h.jdx",
        "3047748b1d286a03c374c0975650c71f80f065d10c71d780cde40c6ac265dd8f",
    )
    original = interop.from_interop
    converted = []

    def record(result, **kwargs):
        dataset = original(result, **kwargs)
        converted.append(dataset)
        return dataset

    monkeypatch.setattr(interop, "from_interop", record)
    dataset = scp.nmr.read(path, protocol="interop", profile="jcampdx-spectrum-1d")
    assert len(converted) == 1
    assert dataset is converted[0]
    assert dataset.shape == (104858,)
    np.testing.assert_array_equal(dataset.data, jcampdx.read(path)[1])
    assert dataset.x.units == scp.ur.Hz
    np.testing.assert_allclose(
        dataset.x.data[[0, -1]], [4996.512910356473, -998.6909265739829]
    )
    assert dataset.meta.data_domain == "frequency"


def test_public_jcampdx_2d():
    from nmrglue import jcampdx

    path = reference(
        "betapinene_cosy_400mhz.jdx",
        "417dafe5122892d49797728c06176f003a7ec17ee31f5060e7bc7a06ee64a2d2",
    )
    dataset = scp.nmr.read(path, protocol="interop", profile="jcampdx-spectrum-2d")
    assert dataset.shape == (128, 3278)
    assert dataset.meta.axis_roles == ("F1", "F2")
    assert dataset.x.units == dataset.y.units == scp.ur.Hz
    np.testing.assert_array_equal(
        dataset.data, np.stack(jcampdx.read(path, show_all_data=True)[1]["real"])
    )
    np.testing.assert_allclose(
        dataset.y.data[[0, 1, 2, -1]], [4997.59, 4950.37, 4903.14, -999.77]
    )
    np.testing.assert_allclose(dataset.x.data[[0, -1]], [5003.72, -1005.9])


def test_public_jcampdx_fid(tmp_path):
    path = tmp_path / "fid.dx"
    path.write_text(
        """##TITLE=complex FID
##JCAMP-DX=5.01
##DATA TYPE=NMR FID
##DATA CLASS=NTUPLES
##NTUPLES=NMR FID
##SYMBOL=X,R,I
##UNITS=SECONDS,ARBITRARY UNITS,ARBITRARY UNITS
##VAR_DIM=3,3,3
##FIRST=0.001,2,10
##LAST=0.003,6,14
##FACTOR=0.001,2,0.5
##PAGE=N=1
##DATA TABLE=(X++(I..I)),XYDATA
1 20 24 28
##PAGE=N=2
##DATA TABLE=(X++(R..R)),XYDATA
1 1 2 3
##END NTUPLES=NMR FID
##END=
"""
    )
    dataset = scp.nmr.read(
        path,
        protocol="interop",
        profile="jcampdx-fid-1d",
        normalize_options={"conjugation": "conjugate"},
    )
    np.testing.assert_array_equal(dataset.data, [2 - 10j, 4 - 12j, 6 - 14j])
    np.testing.assert_allclose(dataset.x.data, [0.001, 0.002, 0.003])
    assert dataset.x.units == scp.ur.s


def test_public_nmrml_2d():
    pytest.importorskip("xmltodict")
    path = reference(
        "synthetic-2d-real.nmrML",
        "60cb45948fdfceef095596bda61b2efaf6a2f3f46c1cdef33019a1b5b339023c",
    )
    dataset = scp.nmr.read(
        path,
        protocol="interop",
        profile="nmrml-2d-spectrum",
        read_options={
            "shape": (3, 4),
            "indirect_axis": (-100, 100, "Hz"),
            "storage_order": "x-fastest",
            "axis_endpoint": "last-point",
        },
    )
    np.testing.assert_array_equal(
        dataset.data, [[1, -2, 4, 8], [16, -32, 64, 128], [3, 5, -7, 11]]
    )
    np.testing.assert_array_equal(dataset.y.data, [-100, 0, 100])
    np.testing.assert_array_equal(dataset.x.data, [9, 7, 5, 3])
    assert dataset.y.units == scp.ur.Hz
    assert dataset.x.units == scp.ur.ppm


def test_public_interop_rejects_unknown_source_and_multiple_paths(tmp_path):
    path = tmp_path / "unknown.ft2"
    path.write_bytes(b"unrecognized")
    with pytest.raises(ValueError, match="Cannot detect an interop profile"):
        scp.nmr.read(path, protocol="interop")
    with pytest.raises(TypeError, match="exactly one path"):
        scp.nmr.read(path, path, protocol="interop")
