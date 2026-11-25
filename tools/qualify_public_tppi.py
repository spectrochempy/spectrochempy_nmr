"""Read-only TPPI/QSEQ JCAMP qualification against a Bruker/TopSpin copy.

External acquisitions and TopSpin exports are supplied separately; no download or
vendor processing is performed. See docs/reference/interop.rst for provenance,
export commands, processing settings and scientific limits.
"""

import argparse
import hashlib
import json
from pathlib import Path

import nmrglue as ng
import numpy as np

from spectrochempy_nmr import Experiment, read


def _native_raw_scale(acqus):
    """Convert the qualified native integer NC to a reference-only scale."""
    nc = acqus.get("NC")
    dtype = acqus["DTYPA"]
    if (
        isinstance(nc, bool)
        or not isinstance(nc, int)
        or not -1022 <= nc <= 1023
        or dtype not in (0, 2)
        or (dtype == 2 and nc != 0)
    ):
        raise ValueError(
            "Qualification requires int32 NC scaling or unscaled float64 raw data"
        )
    return nc, 2.0**nc


def qualify(
    acquisition, processed, exports, prefix, *, digital_filter="none", quadrature="tppi"
):
    """Check raw codecs, four FFT planes, optional phase correction, ppm and RR."""
    if digital_filter not in ("none", "phase"):
        raise ValueError("Qualification digital_filter must be 'none' or 'phase'")
    if quadrature not in ("tppi", "qseq"):
        raise ValueError("Qualification quadrature must be tppi or qseq")
    paths = [acquisition / name for name in ("ser", "acqus", "acqu2s")]
    paths += [
        processed / name for name in ("procs", "proc2s", "2rr", "2ri", "2ir", "2ii")
    ]
    paths += [
        exports / f"{prefix}-{codec}.dx"
        for codec in ("FIX", "PACKED", "SQUEEZED", "DIFF_DUP", "processed")
    ]
    before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }
    for name in ("ser", "acqus", "acqu2s"):
        copied = processed.parent.parent / name
        if copied.read_bytes() != (acquisition / name).read_bytes():
            raise ValueError(f"TopSpin copy changed acquisition input: {name}")
    status, raw = ng.bruker.read(acquisition, read_pulseprogram=False)
    if status["acqu2s"]["FnMODE"] != {"tppi": 3, "qseq": 2}[quadrature] or status[
        "acqus"
    ]["AQ_mod"] not in (1, 3):
        raise ValueError(
            "Qualification requires matching TPPI/QSEQ F1 and QSIM/DQD F2 metadata"
        )
    nc, raw_scale = _native_raw_scale(status["acqus"])
    meta, native = ng.bruker.read_pdata(
        processed, bin_files=["2rr", "2ri", "2ir", "2ii"], scale_data=True
    )
    for key, size, td, ft in (
        ("proc2s", raw.shape[0], raw.shape[0], 5 if quadrature == "tppi" else 7),
        ("procs", raw.shape[1], 2 * raw.shape[1], 6),
    ):
        axis = meta[key]
        for name, expected in {
            "SI": size,
            "TDeff": td,
            "FT_mod": ft,
            "WDW": 0,
            "BC_mod": 0,
            "ME_mod": 0,
            "FCOR": 1,
            "PHC0": 0,
            "PHC1": 0,
            "PKNL": key == "procs" and digital_filter == "phase",
            "STSR": 0,
            "STSI": size,
            "TDoff": 0,
            "REVERSE": False,
        }.items():
            if axis[name] != expected:
                raise ValueError(f"Unsupported reference processing: {key}.{name}")
    for codec in ("FIX", "PACKED", "SQUEEZED", "DIFF_DUP"):
        stored = ng.interop.read(
            exports / f"{prefix}-{codec}.dx", profile="jcampdx-fid-2d"
        )
        np.testing.assert_array_equal(stored.data, raw)
    source = read(
        exports / f"{prefix}-FIX.dx",
        protocol="interop",
        profile="jcampdx-fid-2d",
        normalize_options={
            "quadrature": quadrature,
            "conjugation": "conjugate",
            "indirect_dwell_s": 1 / (2 * status["acqu2s"]["SW_h"]),
        },
    )
    anchors = {
        role: {
            "reference_frequency_mhz": meta[key]["SF"],
            "point_index": 0,
            "ppm": meta[key]["OFFSET"],
            "source": "processing",
        }
        for role, key in (("F1", "proc2s"), ("F2", "procs"))
    }
    original = source.data.copy()
    result = Experiment(source).process(
        digital_filter=digital_filter, reference=anchors
    )
    np.testing.assert_array_equal(source.data, original)
    planes = np.stack(
        (result.data[0], -result.data[2], -result.data[1], result.data[3])
    )
    native = np.asarray(native) / raw_scale
    ny, nx = raw.shape
    planes[0, ny // 2, nx // 2] = (
        planes[0, ny // 2 - 1, nx // 2] + planes[0, ny // 2 + 1, nx // 2]
    ) / 2
    errors = np.linalg.norm(planes - native, axis=(1, 2)) / np.linalg.norm(
        native, axis=(1, 2)
    )
    if not np.isfinite(errors).all() or max(errors) >= 2e-6:
        raise AssertionError(f"Signed-plane relative L2 errors: {errors}")
    for dim, key in (("y", "proc2s"), ("x", "procs")):
        axis = meta[key]
        ppm = axis["OFFSET"] - np.arange(axis["SI"]) * axis["SW_p"] / (
            axis["SF"] * axis["SI"]
        )
        np.testing.assert_allclose(result.coordset[dim].data, ppm, rtol=0, atol=1e-9)
    stored = ng.interop.read(
        exports / f"{prefix}-processed.dx", profile="jcampdx-spectrum-2d"
    )
    _, rr = ng.bruker.read_pdata(processed, scale_data=False)
    np.testing.assert_array_equal(stored.data, rr)
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if before != after:
        raise AssertionError("Qualification modified its external inputs")
    return {
        "shape": list(raw.shape),
        "quadrature": quadrature,
        "direct_acquisition_mode": status["acqus"]["AQ_mod"],
        "native_raw_nc": nc,
        "native_raw_scale": raw_scale,
        "relative_l2_topspin_rr_ri_ir_ii": errors.tolist(),
        "raw_codecs_exact": 4,
        "processed_rr_exact": True,
        "ppm_axes": True,
        "filter_correction_applied": digital_filter == "phase",
        "digital_filter": digital_filter,
        "group_delay_points": status["acqus"]["GRPDLY"],
        "intensity_scale": "JCAMP sample units; TopSpin NC_proc and raw NC accounted explicitly in reference only; no fitted scale or scan/gain division",
        "input_sha256": before,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("acquisition", type=Path)
    parser.add_argument("processed", type=Path)
    parser.add_argument("exports", type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--digital-filter", choices=("none", "phase"), default="none")
    parser.add_argument("--quadrature", choices=("tppi", "qseq"), default="tppi")
    args = parser.parse_args()
    print(
        json.dumps(
            qualify(
                args.acquisition,
                args.processed,
                args.exports,
                args.prefix,
                digital_filter=args.digital_filter,
                quadrature=args.quadrature,
            ),
            indent=2,
        )
    )
