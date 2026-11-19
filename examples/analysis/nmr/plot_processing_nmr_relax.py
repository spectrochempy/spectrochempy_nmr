"""
Fit a saturation-recovery T1 curve
===================================

Process the direct axis, integrate one resonance and fit its recovery.
See :doc:`/userguide/relaxation_t1` for the delay coordinate, phase,
model assumptions, units and residual diagnostics.
"""

import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/relax/100/ser"
raw = scp.nmr.read(path, use_list="vdlist", remove_digital_filter=True)
spectrum = raw.em(lb=15 * scp.ur.Hz, dim="x").fft(dim="x")
phase = scp.nmr.apk(spectrum[-1].squeeze()[-140.0:-80.0], order=0, apply=False)
if not phase.estimate.success:
    raise RuntimeError(phase.estimate.message)
spectrum = spectrum.pk(phc0=phase.estimate.p0 * scp.ur.deg, phc1=0 * scp.ur.deg)
signal = spectrum[:, -120.0:-85.0].real.simpson(dim="x")


def recovery(t, I0, T1):
    return I0 * (1 - scp.exp(-t / T1))


fit = scp.Optimize(method="leastsq", usermodels={"recovery": recovery})
fit.script = """
MODEL: T1
shape: recovery
  $ I0: 4000.0, 1.0, none
  $ T1: 2.0, 0.1, none
"""
_ = fit.fit(signal)
print(fit.script)
ax = signal.plot_scatter(marker="o", label="Measured area", ylabel="Integrated area")
_ = fit.predict().plot(ax=ax, clear=False, label="Recovery fit")
_ = ax.legend()
