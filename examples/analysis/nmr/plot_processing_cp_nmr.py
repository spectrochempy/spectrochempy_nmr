"""
Fit a cross-polarization buildup curve
======================================

Read the contact-time series, apply one common processing recipe, integrate
one band and fit its buildup and decay. See :doc:`/userguide/cp_buildup`
for the phase, baseline, units and comparison with two contributions.
"""

import spectrochempy as scp

root = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/CP"
fids = [scp.nmr.read(root / str(n) / "fid") for n in range(80, 95)]
fids.sort(key=lambda fid: fid.meta.p[0][15].to("ms").magnitude)
times = scp.Coord(
    [fid.meta.p[0][15].to("ms").magnitude for fid in fids],
    units="ms",
    title="CP contact time",
)
processed = [
    scp.nmr.Experiment(fid).process(
        apodization="em",
        lb=50.0,
        size=8192,
        digital_filter="pknl",
        phase="manual",
        phc0=37.365900,
        phc1=-205.449468,
    )
    for fid in fids
]
spectra = scp.NDDataset(
    [row.real.data for row in processed],
    coordset=[times, processed[0].x],
    units=processed[0].units,
)
signal = spectra[:, 62.0:79.0].simpson(dim="x")


def cp_buildup(t, amplitude, buildup, decay):
    return amplitude * (1 - scp.exp(-t / buildup)) * scp.exp(-t / decay)


fit = scp.Optimize(method="leastsq", usermodels={"cp_buildup": cp_buildup})
fit.script = """
MODEL: CP
shape: cp_buildup
  $ amplitude: 75.0, 0.0, none
  $ buildup: 1.0, 0.01, none
  $ decay: 10.0, 0.01, none
"""
_ = fit.fit(signal)
print(fit.script)
ax = signal.plot_scatter(marker="o", label="Measured area", ylabel="Integrated area")
_ = fit.predict().plot(ax=ax, clear=False, label="CP fit")
_ = ax.legend()
