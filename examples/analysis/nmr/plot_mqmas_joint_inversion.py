"""
Invert a synthetic MQMAS map with and without MAS
=================================================

Build small physical RF kernels, generate two known populations and compare
nonnegative inversions. Both finite RF pulses use 0-degree phase for a
predominantly absorptive example with overlapping components. The :doc:`guide </userguide/mqmas_inversion_synthetic>`
explains kernel scaling, regularization and interpretation limits.
"""

# %%
import spectrochempy as scp

p = scp.nmr.pulsar
inv = scp.nmr.mqmas_inversion
f1 = scp.Coord.linspace(-16, 15, 32, units="ppm", title="Isotropic F1")
f2 = scp.Coord.linspace(-16, 16, 65, units="ppm", title="F2")
shift = scp.Coord.linspace(-5, 5, 5, units="ppm", title="Isotropic shift")
pq = scp.Coord.linspace(1.2, 1.8, 5, units="MHz", title="Pq")
mqmas = scp.zeros((f1.size, f2.size), coordset=[f1, f2])
mas = scp.zeros(f2.size, coordset=[f2])
settings = dict(
    spin=1.5,
    larmor_hz=130.9e6,
    spinning_hz=20e3,
    spectral_width_hz=24e3,
    points=256,
    accuracy=1,
    sidebands=0,
    max_rf_step_s=1e-6,
)
response = inv.mqmas_response(
    f1.data,
    f2.data,
    (p.Pulse(5e-6, 60e3, 0),),
    (p.Pulse(2.5e-6, 70e3, 0), p.Coherence(1)),
    f1_gaussian_fwhm_hz=900,
    f2_gaussian_fwhm_hz=150,
    **settings,
)
q_kernel = inv.build_kernel(mqmas, shift.data, pq.data, response, asymmetry=0.42)
mas_response = inv.mas_response(
    f2.data,
    (p.IdealPulse(90),),
    gaussian_fwhm_hz=150,
    transitions="central",
    **settings,
)
m_kernel = inv.build_kernel(mas, shift.data, pq.data, mas_response, asymmetry=0.42)
response.cache_clear()

truth = scp.zeros((shift.size, pq.size), coordset=[shift, pq])
truth[1, 1], truth[3, 3] = 0.65, 0.35
mqmas.data = (q_kernel.matrix @ truth.data.ravel()).reshape(mqmas.shape)
mas.data = m_kernel.matrix @ truth.data.ravel()
mas = 1.7 * mas + 0.02 * abs(mas).max()
for seed, spectrum in zip((310, 311), (mqmas, mas), strict=True):
    spectrum += scp.normal(
        scale=0.005 * abs(spectrum).max(),
        size=spectrum.shape,
        seed=seed,
        coordset=spectrum.coordset,
    )

solo = inv.fit(mqmas, q_kernel, regularization=0.001)
joint = inv.fit(mqmas, q_kernel, mas=mas, mas_kernel=m_kernel, regularization=0.001)
maps = [
    truth,
    solo.population / solo.population.sum(),
    joint.population / joint.population.sum(),
]
_ = scp.multiplot(
    maps,
    ncol=3,
    method="contour",
    colorbar=False,
    labels=["Injected fractions", "MQMAS only", "MQMAS + MAS"],
    levels=[0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65],
    vmin=0,
    vmax=0.7,
    cmap="viridis",
    positive_linestyle="-",
    negative_linestyle="--",
    linewidth=0.8,
    xlabel="Pq (MHz)",
    ylabel="Isotropic shift (ppm)",
    figsize=(12, 3.5),
    equal_aspect=False,
)
print(f"MQMAS RMS: {solo.rms:.2%} alone; {joint.rms:.2%} jointly")
print(f"MAS RMS: {joint.mas_rms:.2%}")

_ = scp.plot_multiple(
    [mas, joint.mas_reconstructed, joint.mas_residual],
    labels=["Synthetic MAS", "Joint prediction", "Residual"],
    color=["0.2", "#b2182b", "#2166ac"],
    method="pen",
    legend="best",
    figsize=(8, 3),
)
