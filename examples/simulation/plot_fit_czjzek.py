"""
Fit the disorder width of a synthetic Czjzek spectrum
=====================================================

Fit by direct simulation. Only sigma is varied; the other tensor and
acquisition parameters are fixed. See the
:doc:`guide </userguide/czjzek_fit>` for the objective and convergence checks.
"""

# %%
import spectrochempy as scp

events = (scp.nmr.pulsar.Pulse(0.5e-6, 50e3),)
settings = dict(
    spin=2.5,
    larmor_hz=130e6,
    spectral_width_hz=20e3,
    points=513,
    accuracy=2,
)


def model(sigma_khz):
    return scp.nmr.czjzek.CzjzekDistribution(sigma_khz * 1e3, cq_hz=2e6, eta=0.4)


def processed(spectrum):
    return scp.nmr.pulsar.broaden(spectrum, gaussian_fwhm_hz=400).real


observed = processed(
    scp.nmr.czjzek.simulate_czjzek(
        model(150), events=events, samples=16, seed=7, **settings
    )
)


def objective(parameters, simulate):
    residual = processed(simulate(model(parameters[0]))) - observed
    return float((residual**2).sum()) / float((observed**2).sum())


result = scp.nmr.fit_czjzek(
    objective,
    [110],
    [(80, 220)],
    events=events,
    samples=16,
    seed=7,
    method="Powell",
    options={"xtol": 0.01, "ftol": 1e-6, "maxiter": 12},
    **settings,
)
sigma = result.direct.optimization.x[0]
fitted = processed(
    scp.nmr.czjzek.simulate_czjzek(
        model(sigma), events=events, samples=16, seed=7, **settings
    )
)
print(
    f"Direct fit: sigma = {sigma:.2f} kHz; success = {result.direct.optimization.success}"
)

_ = scp.plot_multiple(
    [observed, fitted, observed - fitted],
    labels=["Synthetic observation", "Direct fit", "Residual"],
    method="pen",
    legend="best",
    figsize=(8, 3),
    xlim=(-3000, 1500),
)
