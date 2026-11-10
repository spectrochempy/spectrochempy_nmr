"""
Follow cross-polarization buildup in a Bruker series
====================================================

Compare how three regions of a solid-state carbon-13 spectrum respond
to the cross-polarization (CP) contact time. The Bruker experiment stores
each of the 15 FIDs in a separate experiment directory.
"""

# %%
# sphinx_gallery_thumbnail_number = 2

from cmath import phase
from math import degrees

import spectrochempy as scp

root = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/CP"
fids = [scp.nmr.read_topspin(root / str(expno) / "fid") for expno in range(80, 95)]


# %%
# Read P15 from each acquisition. In this series, P15 is the contact
# pulse duration. Sort the FIDs by that duration before stacking them.
def contact_ms(fid):
    return fid.meta.p[0][15].to("ms").magnitude


fids.sort(key=contact_ms)
contact_time = scp.Coord(
    [contact_ms(fid) for fid in fids], units="ms", title="CP contact time"
)


# %%
# These acquisitions share a direct-axis grid. Build a pseudo-2D dataset
# with one row per contact time. The contact axis is a series coordinate.
def contact_series(rows):
    return scp.NDDataset(
        [row.data for row in rows],
        coordset=[contact_time, rows[0].x],
        units=rows[0].units,
        title="signal",
    )


series = contact_series(fids)
series

# %%
# Process each FID on its own direct axis so it keeps its acquisition
# metadata through the FFT. Use magnitude spectra for this comparison
# of peak heights across contact times.
complex_spectra = [fid.em(lb=50.0).fft(size=4096) for fid in fids]
spectra = contact_series([abs(spectrum) for spectrum in complex_spectra])
_ = spectra.plot(method="lines", xlim=(190.0, 50.0), colorbar=True, show=False)

# %%
# Follow the maximum in each selected ppm window. These curves illustrate
# the buildup and subsequent decay of the signal in different regions.
windows = [(165.0, 183.0), (91.0, 107.0), (62.0, 79.0)]
curves = [spectra[:, low:high].max(dim="x") for low, high in windows]
_ = scp.plot_multiple(
    curves,
    method="scatter",
    labels=[f"{low:g}–{high:g} ppm" for low, high in windows],
    title="Carbon-13 CP buildup",
    legend="upper right",
    show=False,
)


# %%
# Fit the buildup and decay
# -------------------------
# Use the relatively isolated resonance around 174 ppm. A simple model
# multiplies a rising exponential by a decaying exponential. Both time
# constants are in milliseconds. They describe this peak-height curve
# under the acquisition conditions used here.
def cp_buildup(t, amplitude, buildup, decay):
    return amplitude * (1 - scp.exp(-t / buildup)) * scp.exp(-t / decay)


def fit_buildup(signal, amplitude):
    model = scp.Optimize(method="leastsq")
    model.usermodels = {"cp_buildup": cp_buildup}
    model.script = f"""
MODEL: CP
shape: cp_buildup
  $ amplitude: {amplitude}, 0.0, none
  $ buildup: 1.0, 0.01, none
  $ decay: 10.0, 0.01, none
"""
    return model.fit(signal)


signal = curves[0]
fitter = fit_buildup(signal, amplitude=2.0)
print(fitter.script)

# %%
# Compare the fitted profile with all 15 measurements. The time constants
# are effective estimates from magnitude peak heights. Overlapping peaks
# or a more detailed relaxation model would require a separate analysis.
ax = signal.plot_scatter(marker="o", label="Measured peak height", show=False)
_ = fitter.predict().plot(ax=ax, clear=False, label="Buildup–decay fit", show=False)
_ = ax.legend()

# %%
# Compare with areas after individual phasing
# -------------------------------------------
# The direct FFT uses the phase metadata from each acquisition. Estimate
# only the residual zero-order phase from the complex signal between
# 172 and 177 ppm. Apply it relative to the FFT phase, then integrate the
# real signal over the wider 169–181 ppm interval. This is a local phase
# estimate for one resonance.
phased = []
for spectrum in complex_spectra:
    reference = spectrum[172.0:177.0].trapezoid().data.item()
    correction = -degrees(phase(reference))
    phased.append(spectrum.pk(phc0=correction, rel=True))

local_spectra = contact_series([spectrum[169.0:181.0] for spectrum in phased])
areas = local_spectra.real.trapezoid(dim="x")
dispersion = local_spectra.imag.trapezoid(dim="x")
_ = scp.plot_multiple(
    [areas, dispersion],
    method="scatter",
    labels=["Real area", "Imaginary area"],
    legend="upper right",
    title="Local phasing check around 174 ppm",
    show=False,
)

# %%
# The four shortest contact times give unstable phase estimates. For the
# comparison, use the same 0.8–15 ms interval for both observables and
# scale each to its own maximum. With the early buildup excluded, the
# fitted constants are sensitive to the selected time window.
traces = [signal[0.8:].normalize(dim="y"), areas[0.8:].normalize(dim="y")]
labels = ["Magnitude height", "Phased real area"]
fits = [fit_buildup(trace, amplitude=1.5).predict() for trace in traces]

# The area fit drives the buildup constant far beyond the measured contact
# times. The curves show the comparison, but this fit cannot establish a
# reliable buildup time for the phased area.
ax = scp.plot_multiple(
    traces,
    method="scatter",
    pen=False,
    labels=labels,
    legend="upper right",
    color=["C0", "C1"],
    title="CP kinetics on a common contact-time interval",
    ylabel="Response / maximum",
    show=False,
)
_ = scp.plot_multiple(
    fits,
    method="pen",
    color=["C0", "C1"],
    ax=ax,
    clear=False,
    ylabel="Response / maximum",
    show=False,
)
