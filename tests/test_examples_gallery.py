# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: S101, S603

"""Targeted checks for the public NMR example gallery."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import spectrochempy as scp

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PLUGIN_ROOT / "examples" / "gallery.toml"
VISIBLE_EXAMPLES = [
    entry["path"]
    for entry in tomllib.loads(MANIFEST.read_text(encoding="utf-8"))["examples"]
]
JEOL_DATA = {
    "plot_read_jeol_1d": "epicatechin_1h.jdf",
    "plot_read_jeol_2d": "epicatechin_cosy.jdf",
    "plot_processing_nmr_2d": "epicatechin_cosy.jdf",
}


def _has_nmr_example_data(example: Path) -> bool:
    root = Path(scp.preferences.datadir) / "nmrdata"
    if example.stem == "plot_replay_mqmas_pst5":
        return True  # The example also runs without the optional acquisition.
    if example.stem in JEOL_DATA:
        return (root / "jeol" / JEOL_DATA[example.stem]).is_file()
    if example.stem in {"plot_read_nmrpipe_2d", "plot_analyze_cosy_2d"}:
        return (root / "nmrpipe" / "nmrpipe_2d_cosy_example.ft2").is_file()
    bruker = root / "bruker" / "tests" / "nmr"
    if example.stem == "plot_processing_cp_nmr":
        return all(
            (bruker / "CP" / str(expno) / "fid").is_file() for expno in range(80, 95)
        )
    if example.stem == "plot_processing_nmr_relax":
        return (bruker / "relax" / "100" / "ser").is_file()
    if example.stem == "plot_compare_topspin_1d":
        if not (bruker / "topspin_1d" / "1" / "pdata" / "1" / "1r").is_file():
            return False
    return (bruker / "topspin_1d" / "1" / "fid").is_file()


def test_nmr_gallery_manifest_covers_all_public_examples():
    scripts = {
        str(path.relative_to(PLUGIN_ROOT / "examples"))
        for path in (PLUGIN_ROOT / "examples").rglob("plot_*.py")
    }
    assert len(VISIBLE_EXAMPLES) == len(scripts)
    assert set(VISIBLE_EXAMPLES) == scripts


@pytest.mark.parametrize("relative_path", VISIBLE_EXAMPLES, ids=VISIBLE_EXAMPLES)
def test_public_nmr_gallery_examples_execute(relative_path, tmp_path):
    example = PLUGIN_ROOT / "examples" / relative_path
    if not _has_nmr_example_data(example):
        pytest.skip("NMR example data not available in this environment")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")

    result = subprocess.run(
        [sys.executable, str(example)],
        cwd=PLUGIN_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    error_msg = (
        f"Example failed: {relative_path}\n"
        f"Return code: {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}\n"
    )
    assert result.returncode == 0, error_msg
