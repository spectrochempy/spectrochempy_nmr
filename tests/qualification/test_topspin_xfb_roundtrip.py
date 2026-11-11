"""Qualify saved XFB replay against TopSpin processed references."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr.experiment import Experiment


_CHECKSUMS = {
    "ser": "641707c170f8352785ee3bc923c52f85a335984c38e0ace7c591c054d32ccc6b",
    "acqus": "ed1db82749fac2dd724305260ac02c26ed76bb11b5b6e20d1dc2652ef21f4636",
    "acqu2s": "b3831b60edfa84e06a89137b121958a8c905b37dfbcf0d056ee3318722611989",
    "pdata/1/2rr": "3852aaed6d3fcea61fcb022c73802efd5b62de1e2c41eb6cd1ea3ded29fe28ed",
    "pdata/1/procs": "15083e49296a29e66dfb85c41519680ab4157c0cdebf455f89af9195dec73261",
    "pdata/1/proc2s": "f5f4838b2e39188de217214882c105f3d04a8de81d2fbaba565dab596b8bb5b8",
}

_RELOAD = """
from importlib.abc import MetaPathFinder
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])

class BlockProvider(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "nmrglue" or fullname.startswith("nmrglue."):
            raise ModuleNotFoundError(fullname, name=fullname)

def block_source(event, args):
    if event == "open" and isinstance(args[0], (str, bytes)):
        if Path(os.fsdecode(args[0])).resolve().is_relative_to(root):
            raise PermissionError("original Bruker reference is unavailable")

sys.meta_path.insert(0, BlockProvider())
sys.addaudithook(block_source)

import numpy as np
import spectrochempy as scp
from spectrochempy_nmr.experiment import Experiment

source = scp.read(Path("source.scp"))
assert not source.meta.source_metadata
result = Experiment(source).process(replay="topspin")
np.savez("reloaded.npz", data=result.data, x=result.x.data, y=result.y.data)
trace = result.meta.nmr_processing["scp_processing"]
Path("trace.json").write_text(json.dumps(trace, allow_nan=False), encoding="utf-8")
assert not any(name.split(".")[0] == "nmrglue" for name in sys.modules)
"""


def test_topspin_xfb_roundtrip_without_provider(tmp_path):
    reference = os.environ.get("NMR_TOPSPIN_XFB_QF_REFERENCE")
    if reference is None:
        pytest.skip("set NMR_TOPSPIN_XFB_QF_REFERENCE to exam2d_HH/1")
    root = Path(reference).resolve(strict=True)
    for name, checksum in _CHECKSUMS.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == checksum, name
    _assert_xfb_roundtrip(root, tmp_path, "qf")


@pytest.mark.parametrize(
    "variable", ["NMR_TOPSPIN_XFB_REFERENCES", "NMR_TOPSPIN_XFB_LP_REFERENCES"]
)
@pytest.mark.parametrize(
    ("experiment", "mode"), [("30", "states-tppi"), ("40", "echo-antiecho")]
)
def test_paired_topspin_xfb_roundtrip_without_provider(
    tmp_path, variable, experiment, mode
):
    reference = os.environ.get(variable)
    if reference is None:
        pytest.skip(f"set {variable} to the corresponding TopSpin XFB copies")
    _assert_xfb_roundtrip(
        Path(reference).resolve(strict=True) / experiment, tmp_path, mode
    )


def _assert_xfb_roundtrip(root, tmp_path, mode):
    import nmrglue as ng

    source = scp.read_topspin(
        root / "ser",
        local_only=True,
        remove_digital_filter=False,
        remove_dc_offset=False,
    )
    plan = source.meta.nmr_processing["vendor_profile"]["xfb_plan"]
    assert plan["mode"] == mode
    expected = Experiment(source).process(replay="topspin")
    names = ["2rr"] if mode == "qf" else ["2rr", "2ri", "2ir", "2ii"]
    metadata, reference = ng.bruker.read_pdata(
        root / "pdata" / "1", bin_files=names, scale_data=True
    )
    if mode == "qf":
        planes = expected.data[None]
        reference = reference[None]
    else:
        assert tuple(expected.c.labels) == ("S_R", "S_I")
        first, second = expected.data
        planes = np.stack((first.real, second.real, first.imag, second.imag))
        reference = np.asarray(reference)
    acqus = metadata["acqus"]
    scale = acqus["NS"] * acqus["RG"]
    if acqus["DTYPA"] == 0:
        scale *= 2.0 ** acqus["NC"]
    assert planes.shape == reference.shape
    residual = np.linalg.norm(planes * scale - reference, axis=(1, 2))
    np.testing.assert_array_less(
        residual / np.linalg.norm(reference, axis=(1, 2)), 2e-6
    )
    for axis, key in ((expected.y, "proc2s"), (expected.x, "procs")):
        pars = metadata[key]
        grid = pars["OFFSET"] - np.arange(pars["SI"]) * pars["SW_p"] / (
            pars["SF"] * pars["SI"]
        )
        assert axis.units == scp.ur.ppm
        np.testing.assert_allclose(axis.data, grid, rtol=0, atol=2e-12)

    source.meta.readonly = False
    source.meta.source_metadata = {}
    source.write(tmp_path / "source.scp", confirm=False)
    subprocess.run(
        [sys.executable, "-I", "-c", _RELOAD, str(root)], cwd=tmp_path, check=True
    )

    with np.load(tmp_path / "reloaded.npz") as actual:
        np.testing.assert_array_equal(actual["data"], expected.data)
        np.testing.assert_array_equal(actual["x"], expected.x.data)
        np.testing.assert_array_equal(actual["y"], expected.y.data)
    trace = expected.meta.nmr_processing["scp_processing"]
    assert json.loads(
        (tmp_path / "trace.json").read_text(encoding="utf-8")
    ) == json.loads(json.dumps(trace, allow_nan=False))
