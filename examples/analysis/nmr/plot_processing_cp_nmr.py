"""
Follow cross-polarization buildup in a Bruker series
====================================================

Compare how three regions of a solid-state carbon-13 spectrum respond
to the cross-polarization (CP) contact time. The Bruker experiment stores
each of the 15 FIDs in a separate experiment directory.
"""

# %%
# sphinx_gallery_thumbnail_number = 2


import spectrochempy as scp

root = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/CP"


# P15 is the contact pulse duration stored in each acquisition.
def contact_ms(fid):
    return fid.meta.p[0][15].to("ms").magnitude


acquisitions = [
    (
        expno,
        scp.nmr.read_topspin(
            root / str(expno) / "fid",
            remove_digital_filter=False,
            remove_dc_offset=False,
        ),
    )
    for expno in range(80, 95)
]
acquisitions.sort(key=lambda entry: contact_ms(entry[1]))
fids = [fid for _, fid in acquisitions]
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
# Transform and phase each spectrum
# ---------------------------------
# All FIDs in this series use the same acquisition conditions apart from
# the contact pulse duration. Use a common phase so the processing is
# consistent across contact times. The angles are from TopSpin 4.5.0
# ``apk`` on experiment 86 (1.4 ms), where all three regions are visible.
# They are specified in degrees at the first frequency bin. PKNL reads
# the fractional delay and applies it once on the final FFT grid.
phc0, phc1 = 37.365900, -205.449468
phased = [
    scp.nmr.Experiment(fid).process(
        apodization="em",
        lb=50.0,
        size=8192,
        digital_filter="pknl",
        phase="manual",
        phc0=phc0,
        phc1=phc1,
    )
    for fid in fids
]

# %%
# Correct the baseline and integrate
# ----------------------------------
# Fit a straight line to the same off-peak windows and axis endpoints in each real
# spectrum. This uses SpectroChemPy's baseline correction.
spectra = contact_series([spectrum.real for spectrum in phased])
corrected = spectra.basc(
    (220.0, 245.0), (-110.0, -60.0), model="polynomial", order=1, include_limits=True
)
_ = corrected.plot(method="lines", xlim=(205.0, 35.0), colorbar=True, show=False)

# %%
# Integrate the same three ppm windows at each contact time.
windows = [(165.0, 183.0), (91.0, 107.0), (62.0, 79.0)]
curves = [corrected[:, low:high].simpson(dim="x") for low, high in windows]
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
# Fit the strongest spectral region, 62–79 ppm. A simple model
# multiplies a rising exponential by a decaying exponential. Both
# constants are effective time scales in milliseconds for this series.
def cp_buildup(t, amplitude, buildup, decay):
    return amplitude * (1 - scp.exp(-t / buildup)) * scp.exp(-t / decay)


signal = curves[-1]
fitter = scp.Optimize(method="leastsq")
fitter.usermodels = {"cp_buildup": cp_buildup}
fitter.script = """
MODEL: CP
shape: cp_buildup
  $ amplitude: 75.0, 0.0, none
  $ buildup: 1.0, 0.01, none
  $ decay: 10.0, 0.01, none
"""
_ = fitter.fit(signal)
print(fitter.script)

# %%
ax = signal.plot_scatter(marker="o", label="Measured area", show=False)
_ = fitter.predict().plot(ax=ax, clear=False, label="Buildup–decay fit", show=False)
_ = ax.legend()
