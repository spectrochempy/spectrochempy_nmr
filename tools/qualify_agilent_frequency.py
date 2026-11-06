"""Compare an even-sized real FID with an explicitly configured NMRPipe oracle.

Requires an installed plugin/provider and external NMRPipe. Oracle OBS/CAR are
supplied independently, never taken from the function under test. Original files
are read-only; binary outputs and acquisition copies stay in a temporary folder.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
import nmrglue as ng

from spectrochempy_nmr import read_agilent_interop, reference_agilent_1d


def qualify(acquisition, binaries, reference, observation_mhz, carrier_ppm):
    acquisition, binaries = Path(acquisition), Path(binaries)
    before = {
        name: hashlib.sha256((acquisition / name).read_bytes()).hexdigest()
        for name in ("fid", "procpar")
    }
    raw = read_agilent_interop(acquisition)
    if raw.size % 2:
        raise ValueError("this external oracle qualifies even-sized acquisitions only")
    fft = raw.fft()
    referenced = reference_agilent_1d(fft, reference=reference)
    with tempfile.TemporaryDirectory(prefix="agilent-frequency-oracle-") as location:
        root = Path(location)
        for name in before:
            shutil.copy2(acquisition / name, root / name)
        converted, transformed = root / "converted.fid", root / "transformed.ft"
        subprocess.run(
            [
                str(binaries / "var2pipe"),
                "-in",
                str(root / "fid"),
                "-noaswap",
                "-xN",
                str(2 * raw.size),
                "-xT",
                str(raw.size),
                "-xMODE",
                "Complex",
                "-xSW",
                str(raw.meta.acquisition["spectral_width_hz"]),
                "-xOBS",
                str(observation_mhz),
                "-xCAR",
                str(carrier_ppm),
                "-xLAB",
                str(raw.meta.acquisition["nucleus"]),
                "-ndim",
                "1",
                "-out",
                str(converted),
                "-ov",
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                str(binaries / "nmrPipe"),
                "-in",
                str(converted),
                "-fn",
                "FT",
                "-out",
                str(transformed),
                "-ov",
            ],
            check=True,
            capture_output=True,
        )
        _, decoded = ng.pipe.read(str(converted))
        header, spectrum = ng.pipe.read(str(transformed))
        np.testing.assert_array_equal(decoded, np.conj(raw.data))
        relative = float(
            np.linalg.norm(spectrum - np.conj(fft.data)) / np.linalg.norm(spectrum)
        )
        ppm_error = float(
            np.max(
                np.abs(
                    ng.pipe.make_uc(header, spectrum).ppm_scale() - referenced.x.data
                )
            )
        )
        if not np.isfinite(relative) or relative > 2e-6:
            raise AssertionError(f"NMRPipe complex-spectrum mismatch: {relative}")
        # NMRPipe stores OBS/CAR/header coordinates as float32.
        if not np.isfinite(ppm_error) or ppm_error > 2e-5:
            raise AssertionError(f"NMRPipe ppm-axis mismatch: {ppm_error}")
    after = {
        name: hashlib.sha256((acquisition / name).read_bytes()).hexdigest()
        for name in before
    }
    assert before == after, "original acquisition changed"
    version = subprocess.run(
        [str(binaries / "nmrPipe"), "-help"], capture_output=True, text=True
    )
    match = re.search(r"NMRPipe System Version[^\n]*", version.stdout + version.stderr)
    return {
        "source_sha256": before,
        "nmrpipe_version": match.group(0) if match else "unknown",
        "reference": reference,
        "oracle_observation_mhz": observation_mhz,
        "oracle_carrier_ppm": carrier_ppm,
        "relative_complex_error": relative,
        "maximum_ppm_error": ppm_error,
        "points": raw.size,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("acquisition", type=Path)
    parser.add_argument("--nmrpipe-bin", type=Path, required=True)
    parser.add_argument(
        "--reference", choices=("rfl-rfp", "sfrq-reffrq"), required=True
    )
    parser.add_argument("--observation-mhz", type=float, required=True)
    parser.add_argument("--carrier-ppm", type=float, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            qualify(
                args.acquisition,
                args.nmrpipe_bin,
                args.reference,
                args.observation_mhz,
                args.carrier_ppm,
            ),
            indent=2,
        )
    )
