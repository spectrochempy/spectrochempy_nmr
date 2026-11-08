#!/usr/bin/env python3
"""Generate Rance–Kay checkpoints with the external NMRPipe tools."""

import argparse
import hashlib
import os
from pathlib import Path
import tempfile

import nmrglue as ng
import numpy as np

from regenerate_states_tppi_oracle import _run
from regenerate_states_tppi_oracle import _run_pipeline


def generate(output, binaries, ser, *, digital_filter=False):
    binaries = binaries.resolve()
    output = output.resolve()
    environment = os.environ.copy()
    environment["NMRTXT"] = str(binaries.parent / "nmrtxt")
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        if ser is None:
            ser = work / "synthetic.ser"
            y, x = 8, 128
            sw_y, sw_x = 1000, 4000
            rng = np.random.default_rng(20261108)
            rng.integers(-128, 128, size=(2 * y, 2 * x)).astype(">i4").tofile(ser)
        else:
            ser = ser.resolve()
            if hashlib.sha1(ser.read_bytes()).hexdigest() != (
                "a979ed42e5a28066a476557d713c76ce081541da"
            ):
                raise ValueError("SER does not match the official HSQC reference")
            y, x = 128, 2048
            sw_y, sw_x = 7541.47812971342, 7002.80112044818
        words = np.fromfile(ser, dtype=">i4").reshape(2 * y, 2 * x)
        traces = words[:, ::2] + 1j * words[:, 1::2]

        paired = work / "paired.fid"
        _run(
            [
                str(binaries / "bruk2pipe"),
                "-in",
                str(ser),
                "-out",
                str(paired),
                "-ov",
                "-swap",
                "-AMX",
                *(["-decim", "24", "-dspfvs", "12"] if digital_filter else []),
                "-bad",
                "0",
                "-xN",
                str(2 * x),
                "-xT",
                str(x),
                "-yN",
                str(2 * y),
                "-yT",
                str(y),
                "-xMODE",
                "DQD",
                "-yMODE",
                "Echo-AntiEcho",
                "-xSW",
                str(sw_x),
                "-ySW",
                str(sw_y),
                "-xOBS",
                "499.842349248",
                "-yOBS",
                "125.68830353599",
                "-xCAR",
                "0",
                "-yCAR",
                "0",
                "-xLAB",
                "H1",
                "-yLAB",
                "C13",
                "-ndim",
                "2",
                "-aq2D",
                "States",
            ],
            cwd=work,
            environment=environment,
        )
        nmrpipe = binaries / "nmrPipe"
        f2, tp, final = (work / name for name in ("f2.ft2", "tp.ft2", "final.ft2"))
        for source, target, flags in (
            (paired, f2, ["-fn", "FT", "-df" if digital_filter else "-nodf"]),
            (f2, tp, ["-fn", "TP"]),
            (tp, final, ["-fn", "FT", "-nodf"]),
        ):
            _run_pipeline(nmrpipe, source, target, [flags], environment=environment)
        paired_header, shuffled = ng.pipe.read(str(paired))
        _, direct = ng.pipe.read(str(f2))
        _, cross = ng.pipe.read(str(tp))
        header, spectrum = ng.pipe.read(str(final))
        first, second = spectrum[::2].T, spectrum[1::2].T
        np.savez_compressed(
            output,
            raw=np.stack((traces[::2], traces[1::2])),
            shuffled=np.stack((shuffled[::2], shuffled[1::2])),
            f2=np.stack((direct[::2], direct[1::2])),
            cross=np.stack((cross[::2].T, cross[1::2].T)),
            spectrum=np.stack((first.real, second.real, first.imag, second.imag)),
            hz_f1=ng.pipe.make_uc(header, spectrum, dim=1).hz_scale(),
            hz_f2=ng.pipe.make_uc(header, spectrum, dim=0).hz_scale(),
            **({"group_delay": paired_header["FDDMXVAL"]} if digital_filter else {}),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--nmrpipe-bin", type=Path, required=True)
    parser.add_argument("--hsqc-ser", type=Path)
    parser.add_argument("--digital-filter", action="store_true")
    args = parser.parse_args()
    generate(
        args.output, args.nmrpipe_bin, args.hsqc_ser, digital_filter=args.digital_filter
    )
