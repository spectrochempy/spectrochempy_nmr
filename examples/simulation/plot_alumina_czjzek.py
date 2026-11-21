"""
Compare an alumina spectrum with an illustrative Czjzek mixture
===============================================================

Read the archived processed 27Al spectrum from ``datadir`` and overlay a
two-component model. These parameters are illustrative, not fitted. See the
:doc:`guide </userguide/alumina_czjzek>` for the model and experimental limits.
"""

# %%
import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/alumina_800C/12/pdata/1/1r"
observed = scp.nmr.read(path, proc_axis=True).squeeze().real
observed = observed[100.0:-35.0]
observed = observed / observed.max()

larmor_hz = 130.30630648e6
profiles = []
for sigma_hz, cq_hz, shift_ppm in [(0.9e6, 4.5e6, 13), (1.2e6, 5.5e6, 75)]:
    distribution = scp.nmr.czjzek.CzjzekDistribution(
        sigma_hz, cq_hz=cq_hz, eta=0.5, shift_hz=shift_ppm * 1e-6 * larmor_hz
    )
    spectrum = scp.nmr.czjzek.simulate_czjzek(
        distribution,
        samples=128,
        seed=0,
        spin=2.5,
        larmor_hz=larmor_hz,
        spinning_hz=12e3,
        spectral_width_hz=200e3,
        points=4097,
        accuracy=4,
        transitions="central",
    )
    spectrum = scp.nmr.pulsar.broaden(spectrum, gaussian_fwhm_hz=800)
    spectrum.x.meta["acquisition_frequency"] = larmor_hz * scp.ur.Hz
    spectrum.x.ito("ppm")
    profiles.append(spectrum.real)

mixture = 0.65 * profiles[0] + 0.35 * profiles[1]

_ = scp.plot_multiple(
    [observed, mixture.normalize()],
    labels=["Alumine 800 C, measured", "Illustrative mixture"],
    method="pen",
    legend="best",
    figsize=(8, 3.8),
    xlim=(-35, 100),
    reverse=True,
    xlabel="Chemical shift / ppm",
    ylabel="Peak-normalized intensity",
)
