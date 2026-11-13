"""
Fit a quadrupolar powder line shape with PULSAR
=================================================

Fit a synthetic spin-5/2 central transition using a PULSAR simulation as
a custom model in SpectroChemPy. Compare the fit with the input and inspect
the residuals. See :doc:`/userguide/pulsar` for the frequency grid conventions.
"""

from numpy import interp

import spectrochempy as scp
from spectrochempy.nmr import pulsar

# %%
# Generate a synthetic spectrum with Gaussian broadening. PULSAR reports
# the left edges of its bins. The fit compares intensities at bin centers.
simulation = dict(
    spin=2.5,
    larmor_hz=100e6,
    spectral_width_hz=20e3,
    spinning_hz=8e3,
    points=4096,
    accuracy=8,
)
simulated = pulsar.simulate_central_transition(
    cq_hz=2.6e6, eta=0.55, shift_hz=725, **simulation
)
step_hz = simulated.x.data[1] - simulated.x.data[0]
observed = pulsar.broaden(simulated, gaussian_fwhm_hz=115)
observed.set_coordset(x=simulated.x + (step_hz / 2) * scp.ur.Hz)
observed = observed[-4000.0:3500.0]


def central_transition(x, ampl, cq, eta, shift, fwhm):
    spectrum = pulsar.simulate_central_transition(
        cq_hz=cq, eta=eta, shift_hz=shift, **simulation
    )
    spectrum = pulsar.broaden(spectrum, gaussian_fwhm_hz=fwhm)
    centers = spectrum.x.data + step_hz / 2
    return ampl * interp(x, centers, spectrum.data, left=0, right=0)


fit = scp.Optimize(
    usermodels={"pulsar": central_transition},
    amplitude_mode="area",
)
fit.script = """
MODEL: central
shape: pulsar
$ ampl: 1, 0, 3
$ cq: 2500000, 2000000, 3000000
$ eta: 0.5, 0.05, 0.95
$ shift: 680, 0, 1500
$ fwhm: 100, 30, 220
"""
result = fit.fit(observed).result

for dataset in (observed, result.fitted, result.residuals):
    dataset.x.ito("kHz")

_ = scp.plot_multiple(
    [observed, result.fitted],
    method="pen",
    labels=["Simulated input", "Fit"],
    legend="best",
    ls=["-", "--"],
    figsize=(8, 3),
    xlim=(-4, 3.5),
    ylabel="Intensity per bin",
)
_ = result.residuals.plot(
    method="pen", figsize=(8, 2), xlim=(-4, 3.5), ylabel="Residual"
)

# %%
# The fitted parameters are the amplitude, quadrupolar coupling, asymmetry,
# isotropic shift and Gaussian width. The script expresses frequencies in Hz.
# Inspect ``fit.script`` for the optimized values.
