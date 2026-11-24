"""Read-only qualification on Zenodo 7221753 quinine spectra (CC BY 4.0).

Supply the original ``NMR Data.zip`` separately. Data: P. I. Rubim de Santana,
J. S. F. Diz de Almeida, T. C. C. Franca and J. Junker,
https://doi.org/10.5281/zenodo.7221753 and 10.1155/2022/7490691.
No data are downloaded, modified or redistributed by this tool.

TopSpin's exported relative integrals are a processing comparator, not certified
baseline truth. Recovery measures C(y+s)-C(y) against a known digital addition s
on instrumental y; this does not establish the true areas already present in y.
Parameters, windows and cases are fixed before observing the results. AsLS and
polynomial settings are comparison settings, not an optimized ranking.
"""

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import spectrochempy as scp
from scipy.integrate import trapezoid
from spectrochempy import Baseline


ARCHIVE_MD5 = "2eceea1029bb1c43aa15292cc618ce72"
SAMPLES = (("quinine-s07-1", 29), ("quinine-s06-1", 100), ("quinine-s11-1", 101))
ANCHORS = [(-2.0, -1.0), (10.5, 11.5), (14.0, 15.0)]
CASES = (
    ("narrow_positive", 12.0, 0.003, 0.5),
    ("narrow_negative", 12.0, 0.003, -0.5),
    ("broad_weak", 12.0, 0.1, 0.03),
    ("broad_strong", 12.0, 0.1, 0.5),
    ("overlapping", 7.9, 0.003, 0.5),
)


def area(coordinates, values, limits):
    """Trapezoid area on ascending physical coordinates, including both bounds."""
    x, y = np.asarray(coordinates), np.asarray(values)
    low, high = sorted(limits)
    selected = (x >= low) & (x <= high)
    order = np.argsort(x[selected])
    if order.size < 3:
        raise ValueError("integration window must contain at least three samples")
    return float(trapezoid(y[selected][order], x[selected][order]))


def recovery_metrics(coordinates, injected, increment, limits):
    """Report signed known-area recovery without renormalizing either spectrum."""
    expected = area(coordinates, injected, limits)
    if not np.isfinite(expected) or expected == 0:
        raise ValueError("injected reference area must be finite and nonzero")
    actual = area(coordinates, increment, limits)
    return {
        "expected_area": expected,
        "recovered_area": actual,
        "area_bias_percent": 100 * (actual / expected - 1),
    }


def methods(stride):
    """Keep distribution window and AsLS smoothness approximately physical."""
    result = {"none": None}
    for width in (10, 20, 40):
        result[f"distribution_{width}"] = {
            "model": "distribution",
            "distribution_width": max(1, round(width / stride)),
        }
    result["asls"] = {"model": "asls", "lamb": 1e9 / stride**4, "asymmetry": 0.05}
    result["polynomial"] = {
        "model": "polynomial",
        "order": 2,
        "ranges": ANCHORS,
        "include_limits": False,
    }
    return result


def corrected(source, options):
    if options is None:
        return source.data.copy()
    processor = Baseline(**options).fit(source)
    return processor.corrected.data.copy()


def regions_from_export(text, *, integrals):
    """Read integral values or only exported bounds; do not invent a reference."""
    if integrals:
        rows = [
            tuple(map(float, match))
            for match in re.findall(
                r"^\s*\d+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$", text, re.M
            )
        ]
        if not rows or rows[0][2] <= 0:
            raise ValueError("expected a positive exported integral reference")
        reference = rows[0][2]
        return [(low, high, value / reference) for low, high, value in rows]
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith(("#", "A", "P")):
            continue
        low, high = map(float, line.split()[:2])
        rows.append((low, high, None))
    if not rows:
        raise ValueError("expected exported integration bounds")
    return rows


def qualify(archive):
    """Verify the original archive; return JSON-serializable evidence and metrics."""
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "md5").hexdigest()
    if digest != ARCHIVE_MD5:
        raise ValueError("archive does not match Zenodo 7221753 NMR Data.zip")
    if "distribution" not in Baseline.class_traits()["model"].values:
        raise ValueError("this qualification requires the distribution core model")
    report = {
        "source_doi": "10.5281/zenodo.7221753",
        "archive_md5": digest,
        "limits": "TopSpin comparator and digital recovery; no absolute baseline truth",
        "integrals": [],
        "recovery": [],
        "inputs_sha256": {},
        "unique_real_spectra": 0,
    }
    real_hashes, fid_hashes = set(), set()
    with ZipFile(archive) as bundle, tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        prepared = []
        for name, experiment in SAMPLES:
            sample = f"{name}/{experiment}"
            prefix = f"NMR Data/{sample}/"
            has_integrals = prefix + "pdata/1/integrals.txt" in bundle.namelist()
            export = "pdata/1/integrals.txt" if has_integrals else "pdata/1/intrng"
            for suffix in (
                "fid",
                "acqu",
                "acqus",
                "pdata/1/proc",
                "pdata/1/procs",
                "pdata/1/1r",
                "pdata/1/1i",
                export,
            ):
                member = prefix + suffix
                content = bundle.read(member)
                report["inputs_sha256"][member] = hashlib.sha256(content).hexdigest()
                target = root / sample / suffix
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            real_hash = report["inputs_sha256"][prefix + "pdata/1/1r"]
            fid_hash = report["inputs_sha256"][prefix + "fid"]
            if real_hash in real_hashes or fid_hash in fid_hashes:
                raise ValueError(
                    "qualification requires distinct real spectra and FIDs"
                )
            real_hashes.add(real_hash)
            fid_hashes.add(fid_hash)
            regions = regions_from_export(
                (root / sample / export).read_text(), integrals=has_integrals
            )
            prepared.append((sample, regions, export))
        for sample, regions, export in prepared:
            native = scp.nmr.read(
                root / sample / "pdata/1/1r",
                processed=True,
                proc_axis=True,
                local_only=True,
            ).real
            for stride in (1, 2, 4):
                source = native[::stride].copy()
                x = source.x.data
                reference = area(x, source.data, regions[0][:2])
                amplitude = float(np.max(np.abs(source.data[(x >= 7) & (x <= 9)])))
                for method, options in methods(stride).items():
                    control = corrected(source, options)
                    for number, (low, high, published) in enumerate(regions, start=1):
                        report["integrals"].append(
                            {
                                "sample": sample,
                                "stride": stride,
                                "method": method,
                                "options": options,
                                "region": number,
                                "limits_ppm": [low, high],
                                "published_relative": published,
                                "export": export,
                                "relative_to_uncorrected_first": area(
                                    x, control, (low, high)
                                )
                                / reference,
                                "uncorrected_relative": area(
                                    x, source.data, (low, high)
                                )
                                / reference,
                            }
                        )
                    for name, center, sigma, fraction in CASES:
                        injected = (
                            fraction
                            * amplitude
                            * np.exp(-0.5 * ((x - center) / sigma) ** 2)
                        )
                        modified = source.copy()
                        modified.data = source.data + injected
                        increment = corrected(modified, options) - control
                        report["recovery"].append(
                            {
                                "sample": sample,
                                "stride": stride,
                                "method": method,
                                "options": options,
                                "case": name,
                                **recovery_metrics(
                                    x,
                                    injected,
                                    increment,
                                    (center - 5 * sigma, center + 5 * sigma),
                                ),
                            }
                        )
    report["unique_real_spectra"] = len(real_hashes)
    with archive.open("rb") as stream:
        unchanged = hashlib.file_digest(stream, "md5").hexdigest() == digest
    if not unchanged:
        raise RuntimeError("source archive changed during qualification")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.resolve() == arguments.archive.resolve() or (
        arguments.archive.exists()
        and arguments.output.exists()
        and arguments.output.samefile(arguments.archive)
    ):
        parser.error("output must not replace the source archive")
    arguments.output.write_text(
        json.dumps(qualify(arguments.archive), indent=2, allow_nan=False) + "\n"
    )
