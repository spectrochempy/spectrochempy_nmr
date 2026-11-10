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
# TopSpin 4.5.0 ran em, ft and apk on each FID. These phase angles are
# the recorded PHC0 and PHC1 in degrees, indexed by experiment number.
topspin_phase = {
    80: (34.220954, -191.407329),
    81: (22.005597, -168.488440),
    82: (27.384450, -185.311872),
    83: (31.468902, -194.061317),
    84: (44.378617, -218.808930),
    85: (33.129208, -195.222656),
    86: (37.365900, -205.449468),
    87: (44.279733, -220.074474),
    88: (-39.429728, -46.215869),
    89: (-32.312972, -62.206944),
    90: (43.968901, -219.560617),
    91: (35.500365, -199.617042),
    92: (-40.401538, -43.167721),
    93: (63.161308, -256.668820),
    94: (61.879869, -254.838595),
}

# %%
# PKNL reads the fractional delay from each acquisition and applies it
# once on the final FFT grid. The manual angles reproduce the recorded
# TopSpin phase at the first frequency bin.
phased = [
    scp.nmr.Experiment(fid).process(
        apodization="em",
        lb=50.0,
        size=8192,
        digital_filter="pknl",
        phase="manual",
        phc0=topspin_phase[expno][0],
        phc1=topspin_phase[expno][1],
    )
    for expno, fid in acquisitions
]

# %%
# Check the phase away from the main peak
# ----------------------------------------
# On experiments 88, 89 and 92, the recorded first-order angles leave
# dispersion around 175 ppm. Use the first-order angle of experiment 86
# as a reference for these three spectra, while preserving their phase
# at the main peak near 71 ppm.
for index, (expno, _) in enumerate(acquisitions):
    if expno in (88, 89, 92):
        spectrum = phased[index]
        pivot = int(spectrum.x.loc2index(71.0))
        phase_change = topspin_phase[86][1] - topspin_phase[expno][1]
        phased[index] = spectrum.pk(phc1=phase_change, pivot=pivot, rel=True)

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
