"""Compare normalized Bruker 1D Experiment results with explicit NMRPipe inputs.

This oracle deliberately disables digital-filter correction in both programs.
The correction is qualified separately by the integer-shift time-domain tests.
All generated files live in a disposable directory; originals are hash-checked.
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

from spectrochempy_nmr import from_interop
from spectrochempy_nmr.experiment import Experiment


def qualify(
    acquisition,
    binaries,
    *,
    points,
    storage_points,
    spectral_width,
    observation_mhz,
    carrier_ppm,
    size,
    mode,
    big_endian,
):
    acquisition, binaries = Path(acquisition), Path(binaries)
    before = {
        name: hashlib.sha256((acquisition / name).read_bytes()).hexdigest()
        for name in ("fid", "acqus")
    }
    if size < points or size % 2:
        raise ValueError(
            "the external oracle requires an even FFT size covering all acquired points"
        )
    words = np.frombuffer(
        (acquisition / "fid").read_bytes(), dtype=">i4" if big_endian else "<i4"
    )
    if len(words) != storage_points * 2:
        raise ValueError("declared storage size does not match the original int32 file")
    samples = words[: 2 * points : 2].astype(float) + 1j * words[1 : 2 * points : 2]
    raw = ng.interop.read_bruker_fid(acquisition)
    normalized = ng.interop.normalize_bruker_fid(raw)
    np.testing.assert_array_equal(normalized.data, np.conj(samples))
    result = Experiment(from_interop(normalized)).process(
        size=size, reference="acquisition"
    )
    with tempfile.TemporaryDirectory(prefix="bruker-normalized-oracle-") as location:
        root = Path(location)
        shutil.copy2(acquisition / "fid", root / "fid")
        converted, padded, transformed = (
            root / "converted.fid",
            root / "padded.fid",
            root / "transformed.ft",
        )
        commands = [
            [
                str(binaries / "bruk2pipe"),
                "-in",
                str(root / "fid"),
                "-bad",
                "0",
                "-AMX",
                "-swap" if big_endian else "-noswap",
                "-ext",
                "-xN",
                str(2 * storage_points),
                "-xT",
                str(points),
                "-xMODE",
                mode,
                "-xSW",
                str(spectral_width),
                "-xOBS",
                str(observation_mhz),
                "-xCAR",
                str(carrier_ppm),
                "-xLAB",
                "1H",
                "-ndim",
                "1",
                "-out",
                str(converted),
                "-ov",
            ],
            [
                str(binaries / "nmrPipe"),
                "-in",
                str(converted),
                "-fn",
                "ZF",
                "-size",
                str(size),
                "-out",
                str(padded),
                "-ov",
            ],
            [
                str(binaries / "nmrPipe"),
                "-in",
                str(padded),
                "-fn",
                "FT",
                "-nodmx",
                "-out",
                str(transformed),
                "-ov",
            ],
        ]
        for command in commands:
            subprocess.run(command, check=True, capture_output=True)
        _, decoded = ng.pipe.read(converted)
        np.testing.assert_array_equal(decoded, samples)
        header, spectrum = ng.pipe.read(transformed)
        relative = float(
            np.linalg.norm(spectrum - np.conj(result.data)) / np.linalg.norm(spectrum)
        )
        ppm_error = float(
            np.max(
                np.abs(ng.pipe.make_uc(header, spectrum).ppm_scale() - result.x.data)
            )
        )
        if not np.isfinite(relative) or relative > 2e-6:
            raise AssertionError(f"NMRPipe complex-spectrum mismatch: {relative}")
        if not np.isfinite(ppm_error) or ppm_error > 2e-5:
            raise AssertionError(f"NMRPipe ppm-axis mismatch: {ppm_error}")
    after = {
        name: hashlib.sha256((acquisition / name).read_bytes()).hexdigest()
        for name in before
    }
    assert after == before, "original acquisition changed"
    version = subprocess.run(
        [str(binaries / "nmrPipe"), "-help"], capture_output=True, text=True
    )
    match = re.search(r"NMRPipe System Version[^\n]*", version.stdout + version.stderr)
    return {
        "source_sha256": before,
        "nmrpipe_version": match.group(0) if match else "unknown",
        "digital_filter": "disabled in both paths",
        "acquired_points": points,
        "storage_points": storage_points,
        "fft_size": size,
        "oracle_observation_mhz": observation_mhz,
        "oracle_carrier_ppm": carrier_ppm,
        "relative_complex_error": relative,
        "maximum_ppm_error": ppm_error,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("acquisition", type=Path)
    parser.add_argument("--nmrpipe-bin", type=Path, required=True)
    parser.add_argument("--points", type=int, required=True)
    parser.add_argument("--storage-points", type=int, required=True)
    parser.add_argument("--spectral-width", type=float, required=True)
    parser.add_argument("--observation-mhz", type=float, required=True)
    parser.add_argument("--carrier-ppm", type=float, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--mode", choices=("DQD", "Complex"), required=True)
    parser.add_argument("--big-endian", action="store_true")
    args = vars(parser.parse_args())
    args["binaries"] = args.pop("nmrpipe_bin")
    print(json.dumps(qualify(**args), indent=2))


if __name__ == "__main__":
    main()
