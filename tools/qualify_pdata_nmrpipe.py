"""Compare preserved int32 pdata components and processing axes with NMRPipe.

Oracle size, SF, OFFSET, SW_p and NC_proc are explicit inputs. Conversion imports
an already-frequency-domain real component, supplies its last-point origin, and
never runs FFT, phase or acquisition corrections. Originals remain read-only.
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
    spectral_width,
    reference_mhz,
    offset_ppm,
    nc_proc,
    big_endian=False,
):
    acquisition, binaries = Path(acquisition), Path(binaries)
    names = [name for name in ("1r", "1i", "procs") if (acquisition / name).is_file()]
    before = {
        name: hashlib.sha256((acquisition / name).read_bytes()).hexdigest()
        for name in names
    }
    if points < 2 or reference_mhz <= 0 or spectral_width <= 0:
        raise ValueError("oracle size, frequency and spectral width must be positive")
    raw = ng.interop.read_bruker_pdata(acquisition, all_components=True)
    normalized = ng.interop.normalize_bruker_pdata(raw)
    dataset = from_interop(normalized)
    result = Experiment(dataset).process()
    if result.size != points:
        raise ValueError("oracle point count disagrees with source")
    origin = offset_ppm * reference_mhz - (points - 1) * spectral_width / points
    carrier = offset_ppm - spectral_width / (2 * reference_mhz)
    errors = {}
    maximum_ppm_error = 0.0
    with tempfile.TemporaryDirectory(prefix="pdata-nmrpipe-oracle-") as location:
        root = Path(location)
        for name in ("1r", "1i"):
            if name not in before:
                continue
            if (acquisition / name).stat().st_size != points * 4:
                raise ValueError(
                    "the external oracle requires complete int32 components"
                )
            shutil.copy2(acquisition / name, root / name)
            output = root / (name + ".ft")
            command = [
                str(binaries / "bruk2pipe"),
                "-in",
                str(root / name),
                "-bad",
                "0",
                "-AMX",
                "-swap" if big_endian else "-noswap",
                "-xN",
                str(points),
                "-xT",
                str(points),
                "-xMODE",
                "Real",
                "-xFT",
                "Freq",
                "-xORIG",
                str(origin),
                "-xSW",
                str(spectral_width),
                "-xOBS",
                str(reference_mhz),
                "-xCAR",
                str(carrier),
                "-xLAB",
                "1H",
                "-ndim",
                "1",
                "-out",
                str(output),
                "-ov",
            ]
            subprocess.run(command, check=True, capture_output=True)
            header, values = ng.pipe.read(output)
            assert header["FDF2FTFLAG"] == 1.0
            component = result.data.real if name == "1r" else result.data.imag
            difference = float(np.linalg.norm(values * (2.0**nc_proc) - component))
            norm = float(np.linalg.norm(component))
            error = (
                difference / norm
                if norm
                else (0.0 if difference == 0 else float("inf"))
            )
            axis_error = float(
                np.max(
                    np.abs(
                        ng.pipe.make_uc(header, values).ppm_scale()
                        - result.x.to("ppm").data
                    )
                )
            )
            if not np.isfinite(error) or error > 2e-6:
                raise AssertionError(
                    f"NMRPipe processed component mismatch: {name}, {error}"
                )
            if not np.isfinite(axis_error) or axis_error > 2e-5:
                raise AssertionError(f"NMRPipe processed ppm mismatch: {axis_error}")
            errors[name] = error
            maximum_ppm_error = max(maximum_ppm_error, axis_error)
    assert before == {
        name: hashlib.sha256((acquisition / name).read_bytes()).hexdigest()
        for name in names
    }
    version = subprocess.run(
        [str(binaries / "nmrPipe"), "-help"], capture_output=True, text=True
    )
    match = re.search(r"NMRPipe System Version[^\n]*", version.stdout + version.stderr)
    return {
        "source_sha256": before,
        "nmrpipe_version": match.group(0) if match else "unknown",
        "oracle_reference_mhz": reference_mhz,
        "oracle_offset_ppm": offset_ppm,
        "oracle_spectral_width_hz": spectral_width,
        "oracle_nc_proc": nc_proc,
        "last_point_origin_hz": origin,
        "relative_component_errors": errors,
        "maximum_ppm_error": maximum_ppm_error,
        "points": points,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("acquisition", type=Path)
    parser.add_argument("--nmrpipe-bin", dest="binaries", type=Path, required=True)
    parser.add_argument("--points", type=int, required=True)
    parser.add_argument("--spectral-width", type=float, required=True)
    parser.add_argument("--reference-mhz", type=float, required=True)
    parser.add_argument("--offset-ppm", type=float, required=True)
    parser.add_argument("--nc-proc", type=int, required=True)
    parser.add_argument("--big-endian", action="store_true")
    print(json.dumps(qualify(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
