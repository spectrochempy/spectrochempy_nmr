"""Compare a Bruker diffSte series with its TopSpin diffusion fit.

Run with the path to the spectrochempy_data acquisition
nmrdata/bruker/tests/nmr/relax/102.
"""

import argparse
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

import spectrochempy as scp

REFERENCE_D = 7.559092e-12  # m²/s, first FITFKT entry in TopSpin t1t2.dx


def diffusion_weights(acquisition):
    """Reproduce the effective rectangular-gradient convention in this export."""
    root = ElementTree.parse(acquisition / "diff.xml").getroot()
    if root.findtext("pulprog") != "diffSte":
        raise ValueError("expected the diffSte qualification acquisition")
    rows = root.findall("./XValues/X")
    np.testing.assert_array_equal([int(row.attrib["i"]) for row in rows], np.arange(12))
    gradients = np.loadtxt(acquisition / "difflist")
    np.testing.assert_allclose(
        gradients, [float(row.attrib["g"]) for row in rows], atol=0.005, rtol=0
    )
    gamma_hz_per_gauss = 1070.58392594692  # Gamma in pdata/1/dosy
    delta_s = float(root.findtext("delta")) * 1e-3
    separation_s = float(root.findtext("DELTA")) * 1e-3
    gamma_rad_per_tesla = 2 * np.pi * gamma_hz_per_gauss * 1e4
    gradient_tesla_per_m = gradients * 1e-2  # G/cm to T/m
    b = (gamma_rad_per_tesla * gradient_tesla_per_m * delta_s) ** 2 * (
        separation_s - delta_s / 3
    )
    exported = np.array([float(row.attrib["B"]) for row in rows])
    np.testing.assert_allclose(b, exported, rtol=1e-5)
    return b, exported, np.max(np.abs(b / exported - 1))


def exported_integrals(acquisition, region=1, *, export=None):
    """Read one XYPOINTS block of this TopSpin JCAMP export."""
    report = (export or acquisition / "pdata/1/t1t2.dx").read_text()
    block = report.split("##XYPOINTS= (XY..XY)\n")[region].split("##END=", 1)[0]
    gradients, areas = np.loadtxt(block.splitlines(), delimiter=",", unpack=True)
    np.testing.assert_allclose(
        gradients, np.loadtxt(acquisition / "difflist"), rtol=5e-4, atol=0.05
    )
    return areas


def raw_integral_variants(acquisition, b):
    """Vary one processing choice at a time for the first integration window."""
    series = scp.nmr.read(acquisition / "ser", remove_digital_filter=True)
    if series is None or series.ndim != 2 or series.shape[0] != b.size:
        raise ValueError("expected one SER trace per gradient")
    # The indirect dimension indexes gradients, not evolution times.
    series.y = scp.Coord(b, units="s/m^2", title="diffusion weighting")
    phc0, phc1 = series.meta.phc0[-1], series.meta.phc1[-1]
    fft = series.fft(dim="x", size=65536)
    phased = fft.pk(dim="x", phc0=phc0, phc1=phc1).real
    reported = slice(72.916, 72.649)

    def integrate(spectrum, window=reported):
        return spectrum[:, window].simpson(dim="x").data

    proc = acquisition / "pdata/1"
    high, low = np.loadtxt(proc / "intrng", skiprows=2, usecols=(0, 1))[0]
    procs = dict(
        line[3:].split("=", 1)
        for line in (proc / "procs").read_text().splitlines()
        if line.startswith("##$")
    )
    ppm = float(procs["OFFSET"]) - np.arange(fft.shape[-1]) * (
        float(procs["SW_p"]) / (float(procs["SF"]) * int(procs["SI"]))
    )
    exact_bins = np.flatnonzero((low <= ppm) & (ppm <= high))
    top_window = slice(exact_bins[0], exact_bins[-1] + 1)

    # Use two adjacent strips to estimate a local offset for each trace.
    # This is a sensitivity test. TopSpin's saved bias and slope are zero.
    x = phased.x.data
    left = (low - 0.08 <= x) & (x < low - 0.025)
    right = (high + 0.025 < x) & (x <= high + 0.08)
    offset = (
        phased.data[:, left].mean(axis=1) + phased.data[:, right].mean(axis=1)
    ) / 2
    baseline = phased.copy()
    baseline.data = phased.data - offset[:, None]

    delay = float(
        next(
            line.partition("=")[2]
            for line in (acquisition / "acqus").read_text().splitlines()
            if line.startswith("##$GRPDLY=")
        )
    )
    # The reader floors GRPDLY. On the centered DQD FFT grid the remaining
    # delay contributes 360 * fraction * (k / N - 1/2) degrees.
    fraction = delay - np.floor(delay)
    residual_delay = fft.pk(
        dim="x",
        phc0=phc0 - 180 * fraction * scp.ur.degree,
        phc1=phc1 + 360 * fraction * scp.ur.degree,
    ).real

    # Equal bin width cancels when each series is normalized to its first trace.
    # Compare strict containment with rounding both endpoints to the nearest bin.
    region_areas = []
    for upper, lower in np.loadtxt(proc / "intrng", skiprows=2, usecols=(0, 1)):
        inside = np.flatnonzero((lower <= ppm) & (ppm <= upper))
        start = np.argmin(np.abs(ppm - upper))
        stop = np.argmin(np.abs(ppm - lower))
        region_areas.append(
            (
                residual_delay.data[:, inside].sum(axis=1),
                residual_delay.data[:, start : stop + 1].sum(axis=1),
                (start, stop),
            )
        )

    selected = residual_delay[:, reported]
    variants = {
        "Stored phase, reported bounds": integrate(phased),
        "Residual fractional group delay": integrate(residual_delay),
        "Residual delay, same bins, sum": selected.data.sum(axis=1),
        "Exact bounds, SCP axis": integrate(phased, slice(float(high), float(low))),
        "Exact bounds, TopSpin axis": integrate(phased, top_window),
        "Bounds widened by 0.02 ppm": integrate(phased, slice(72.936, 72.629)),
        "Bounds narrowed by 0.02 ppm": integrate(phased, slice(72.896, 72.669)),
        "Local baseline offset": integrate(baseline),
        "Phase PHC0 - 5 deg": integrate(
            fft.pk(dim="x", phc0=phc0 - 5 * scp.ur.degree, phc1=phc1).real
        ),
        "Phase PHC1 - 10 deg": integrate(
            fft.pk(dim="x", phc0=phc0, phc1=phc1 - 10 * scp.ur.degree).real
        ),
        "Phase PHC0 - 15 deg": integrate(
            fft.pk(dim="x", phc0=phc0 - 15 * scp.ur.degree, phc1=phc1).real
        ),
    }

    return variants, region_areas


def fit_diffusion(b, areas):
    """Return D in m²/s and residual RMS relative to fitted I0."""

    # Scale the fitted parameter so the optimizer works with values near one.
    def decay(x, ampl, d):
        return ampl * np.exp(-x * d * 1e-12)

    data = scp.NDDataset(areas, coordset=[scp.Coord(b, units="s/m^2")])
    optimizer = scp.Optimize(usermodels={"decay": decay})
    optimizer.script = f"""
MODEL: diffusion
shape: decay
$ ampl: {areas[0]}, 0, none
$ d: 7, 0, none
"""
    result = optimizer.fit(data).result
    if not result.diagnostics["success"]:
        raise RuntimeError(result.diagnostics["message"])
    d = optimizer.fp["d_diffusion"] * 1e-12
    relative_rms = result.diagnostics["rmse"] / optimizer.fp["ampl_diffusion"]
    return d, relative_rms


def compare_reprocessed_export(acquisition, export, b):
    """Compare the first window of two exports on the same gradient grid."""
    historical = exported_integrals(acquisition)
    replayed = exported_integrals(acquisition, export=export)
    historical_d, _ = fit_diffusion(b, historical)
    replayed_d, _ = fit_diffusion(b, replayed)
    historical = historical / historical[0]
    replayed = replayed / replayed[0]
    print("Trace    Historical     Reprocessed    Difference")
    for index, (old, new) in enumerate(zip(historical, replayed, strict=True), 1):
        print(f"{index:5d}   {old:+.8f}    {new:+.8f}    {new - old:+.8f}")
    print(
        f"Normalized area RMS difference: {np.sqrt(np.mean((replayed - historical) ** 2)):.6%}"
    )
    print(f"D historical:  {historical_d:.8g} m²/s")
    print(f"D reprocessed: {replayed_d:.8g} m²/s")
    print(f"Relative D difference: {replayed_d / historical_d - 1:+.6%}")


def report_trace_sensitivity(b, historical, reconstructed):
    """Refit both series on the same b grid, omitting each trace in turn."""
    historical = historical / historical[0]
    reconstructed = reconstructed / reconstructed[0]
    print("\nOmitted trace    Relative D difference")
    for omitted in range(-1, b.size):
        keep = np.arange(b.size) != omitted
        reference_d, _ = fit_diffusion(b[keep], historical[keep])
        reconstructed_d, _ = fit_diffusion(b[keep], reconstructed[keep])
        label = "none" if omitted < 0 else str(omitted + 1)
        print(f"{label:>13s}    {reconstructed_d / reference_d - 1:+.3%}")


def main():
    """Run the export comparison and report the independent SER result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("acquisition", type=Path)
    parser.add_argument(
        "--reprocessed-export",
        type=Path,
        help="Compare a new TopSpin t1t2.dx for the same integration window",
    )
    args = parser.parse_args()
    acquisition = args.acquisition.resolve()
    b, exported_b, b_error = diffusion_weights(acquisition)
    if args.reprocessed_export is not None:
        compare_reprocessed_export(acquisition, args.reprocessed_export, exported_b)
        return
    exported = exported_integrals(acquisition)
    exported_d, exported_rms = fit_diffusion(exported_b, exported)
    np.testing.assert_allclose(exported_d, REFERENCE_D, rtol=2e-4, atol=0)
    print(f"Maximum relative b error: {b_error:.3g}")
    print(f"TopSpin reference:       D = {REFERENCE_D:.8g} m²/s")
    print(
        f"Exported areas:         D = {exported_d:.8g} m²/s, RMS/I0 = {exported_rms:.2%}"
    )
    reference = exported / exported[0]
    variants, region_areas = raw_integral_variants(acquisition, b)
    raw = variants["Stored phase, reported bounds"]
    corrected = variants["Residual fractional group delay"]
    print("\nTrace    TopSpin      SER         SER with residual delay correction")
    values = zip(reference, raw / raw[0], corrected / corrected[0], strict=True)
    for i, (expected, observed, adjusted) in enumerate(values, 1):
        print(f"{i:5d}   {expected:+.6f}   {observed:+.6f}   {adjusted:+.6f}")

    print("\nVariant                         D (1e-12 m²/s)   ΔD (%)   Area RMS (%)")
    for label, areas in variants.items():
        normalized = areas / areas[0]
        d, _ = fit_diffusion(b, normalized)
        rms = np.sqrt(np.mean((normalized - reference) ** 2))
        print(f"{label:31s} {d * 1e12:14.6f} {(d / REFERENCE_D - 1):+9.2%} {rms:12.2%}")

    print("\nRegion    Rule           Nearest bins    ΔD (%)   Area RMS (%)")
    for region, (contained, nearest, bins) in enumerate(region_areas, 1):
        areas = exported_integrals(acquisition, region)
        reference_d, _ = fit_diffusion(exported_b, areas)
        for rule, candidate in (("inside", contained), ("nearest", nearest)):
            fitted_d, _ = fit_diffusion(b, candidate)
            rms = np.sqrt(np.mean((candidate / candidate[0] - areas / areas[0]) ** 2))
            bin_label = f"{bins[0]}–{bins[1]}" if rule == "nearest" else ""
            print(
                f"{region:6d}    {rule:8s} {bin_label:>17s} "
                f"{fitted_d / reference_d - 1:+9.2%} {rms:12.2%}"
            )

    report_trace_sensitivity(
        exported_b, exported, variants["Residual delay, same bins, sum"]
    )


if __name__ == "__main__":
    main()
