# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Invert MQMAS and MAS with a shared distribution
#
# A phased MQMAS map contains overlapping responses with different isotropic
# shifts and quadrupolar products. An inversion estimates nonnegative cell
# weights on that parameter grid. An additional one-dimensional MAS spectrum
# can constrain the same distribution through a different response kernel.
# This guide compares MQMAS alone and the joint fit on a small synthetic
# example with known injected weights.
#
# Both kernels below are calculated with the public PULSAR adapters, rather
# than interchangeable Gaussian peaks. The MQMAS kernel retains finite-RF
# transfer, phase and the spin-dependent isotropic coordinate. The MAS
# kernel uses an ideal 90-degree excitation for this illustrative acquisition.
#
# ## Spectral axes, parameter grid and physical kernels
#
# Spectra require calibrated ppm coordinates. In the 2D input, `y` is F1
# and `x` is F2. Parameter grids must be increasing and uniformly spaced,
# with at least three points per axis. Isotropic shifts are in ppm and
# quadrupolar products are in MHz, with
# `Pq = abs(Cq) * sqrt(1 + eta**2 / 3)`.
#
# This example uses spin 3/2 at 130.9 MHz and 20 kHz MAS. MQ preparation is
# a 5 microsecond, 60 kHz pulse at 0 degrees; conversion is a 2.5 microsecond,
# 70 kHz pulse at 0 degrees, followed by single-quantum pathway selection.
# These RF phases give a predominantly absorptive real response for the
# two injected components. We retain the signed calculation; no absolute
# value or clipping is applied to the spectra or kernels.
# Gaussian FWHM values are 900 Hz in F1 and 150 Hz in F2; the MAS width is
# 150 Hz. The fixed asymmetry is 0.42 for both experiments.
#
# Integer/half-integer ppm steps keep these small coordinate grids exactly
# uniform. Create spectra and population maps with `scp.zeros()` and noise
# with `scp.normal()`. Keep calculations on `NDDataset` objects; `.data` is
# used only when passing coordinates or weights to the numerical kernels
# and assigning their matrix predictions back to the spectra.

# ### Choose the spectral axes and candidate populations
#
# `f1` and `f2` say **where the spectral signal is evaluated**. The MQMAS
# observation has 32 F1 rows and 65 F2 columns; the MAS observation uses
# the same 65-point F2 axis. The zero-filled datasets are templates that
# give the kernel builders their spectral shape and coordinate calibration.
# They do not yet contain simulated signals.
#
# `shift` and `pq` instead define **which populations the fit can assign**:
# 5 shifts times 5 quadrupolar products give 25 candidate cells. These
# parameter axes are independent of the spectral pixel count. Each cell
# will have its own calculated MQMAS and MAS response.

# %%
import spectrochempy as scp

# Short names for the simulation and inversion modules.
p = scp.nmr.pulsar
inv = scp.nmr.mqmas_inversion

# Coordinates of the spectra to be simulated and fitted.
f1 = scp.Coord.linspace(-16, 15, 32, units="ppm", title="Isotropic F1")
f2 = scp.Coord.linspace(-16, 16, 65, units="ppm", title="F2")
mqmas = scp.zeros((f1.size, f2.size), coordset=[f1, f2])
mas = scp.zeros(f2.size, coordset=[f2])

# Coordinates of the unknown population map: shift on y, Pq on x.
shift = scp.Coord.linspace(-5, 5, 5, units="ppm", title="Isotropic shift")
pq = scp.Coord.linspace(1.2, 1.8, 5, units="MHz", title="Pq")

# %% [markdown]
# ### Separate physical conditions from numerical resolution
#
# The first three settings identify the spin, field and MAS rate.
# `spectral_width_hz` and `points` define PULSAR's internal direct-frequency
# grid. That 256-point simulation grid is different from the 65-point F2
# observation grid: the response adapter evaluates the simulated profile
# at the observation coordinates.
#
# `accuracy` controls powder-orientation sampling. `sidebands=0` retains
# the order-zero spinning response. `max_rf_step_s` limits the propagation
# time step during finite RF pulses. These numerical choices affect the
# response calculation; they are not additional populations to fit.

# %%
settings = dict(
    spin=1.5,  # Nuclear spin quantum number, not a frequency.
    larmor_hz=130.9e6,
    spinning_hz=20e3,
    spectral_width_hz=24e3,
    points=256,  # Internal simulation points, before evaluation on f2.
    accuracy=1,  # Coarse powder sampling for this quick demonstration.
    sidebands=0,
    max_rf_step_s=1e-6,
)

# %% [markdown]
# ### Build the MQMAS response and its matrix
#
# `Pulse(duration_s, rf_hz, phase_deg)` describes a finite RF pulse.
# The preparation pulse excites MQ coherence, and the conversion events
# transfer it towards detection. `Coherence(1)` selects the single-quantum
# pathway; the MQMAS model calculates the central-transition readout.
# The phases are RF pulse phases, not fitted spectrum phase corrections.
#
# `mqmas_response()` creates a **callable simulator** for one candidate
# `(shift_ppm, cq_hz, eta)`; it does not create the population map.
# `build_kernel()` then evaluates that simulator for all 25 cells at the
# fixed asymmetry 0.42. The resulting `q_kernel.matrix` has 2080 spectral
# rows (32 times 65) and 25 population columns. Multiplying it by a vector
# of cell weights predicts the flattened MQMAS spectrum.

# %%
preparation = (p.Pulse(5e-6, 60e3, 0),)
conversion = (p.Pulse(2.5e-6, 70e3, 0), p.Coherence(1))
response = inv.mqmas_response(
    f1.data,
    f2.data,
    preparation,
    conversion,
    f1_gaussian_fwhm_hz=900,
    f2_gaussian_fwhm_hz=150,
    **settings,
)
# Calculate one signed MQMAS response column per candidate population cell.
q_kernel = inv.build_kernel(mqmas, shift.data, pq.data, response, asymmetry=0.42)

# %% [markdown]
# ### Build the MAS response on the same population grid
#
# The MAS experiment uses an ideal 90-degree excitation here, followed by
# central-transition detection and 150 Hz Gaussian broadening. Its response
# differs from MQMAS, but the columns still refer to the **same 25 candidate
# populations**. `m_kernel.matrix` has 65 spectral rows and 25 columns.
# This shared column meaning is what makes a joint population fit possible.
#
# Once the matrices exist, the fit reuses them without propagating RF pulses
# again. `cache_clear()` releases the intermediate MQMAS RF profiles; it
# does not erase either kernel matrix.

# %%
mas_response = inv.mas_response(
    f2.data,
    (p.IdealPulse(90),),
    gaussian_fwhm_hz=150,
    transitions="central",
    **settings,
)
# Calculate the complementary MAS response for those same cells.
m_kernel = inv.build_kernel(mas, shift.data, pq.data, mas_response, asymmetry=0.42)
response.cache_clear()

# %% [markdown]
# `build_kernel()` calls the response at every shift/Pq cell. At fixed Pq,
# it converts the specified asymmetry to Cq. With `asymmetry="uniform"` or
# `"czjzek"`, it instead averages the response over eta quadrature nodes.
# Use the same parameter grid and asymmetry assumption for both experiments.
# The two kernels share columns, but their spectral grids can differ.
#
# The RF transfer and signed response scale are retained: do not separately
# normalize every kernel column or take absolute values of phased spectra.
# The matrix rows use C ordering of the spectral axes, while columns enumerate
# shift first and Pq second. Clearing the response cache releases RF profiles
# once the immutable matrices have been constructed.
#
# Accuracy 1, a 5 by 5 parameter grid and 256 direct simulation points keep
# this demonstration short. They are not converged settings for a measured
# spectrum. The fast MQMAS response also uses the fixed-transfer approximation
# of `MQMASModel`; an actual acquisition must match its assumptions and phase,
# shear, carrier and referencing conventions.
#
# ## Inject a known distribution and generate observations
#
# The two cells are at (-2.5 ppm, 1.35 MHz) and (2.5 ppm, 1.65 MHz), with
# weights 0.65 and 0.35. Their separation is chosen to show two overlapping
# positive maxima in MQMAS and a shoulder in MAS. Peak heights do not directly
# give population fractions because the RF responses differ. The MAS
# observation has an arbitrary positive gain
# and constant baseline. Each observation receives independent Gaussian noise
# with standard deviation 0.5% of its clean absolute peak. Fixed seeds 310
# and 311 reproduce the MQMAS and MAS noise, respectively.
#
# Generation and inversion deliberately use the same kernels and on-grid
# parameters. This is a reproducible workflow check, not independent validation
# of the physical model or a realistic off-grid resolution study.
#
# The two panels below share the same F2 axis, from +16 to -16 ppm. The
# lower panel is the separately simulated MAS acquisition for the same
# population; the shared frequency scale makes the band positions comparable.
# Contour magnitudes start at 2% of the maximum and increase by a factor
# of 1.4, giving several contours across weak bands. The viridis palette
# encodes intensity; positive contours are solid and negative ones dashed.
# At these RF phases the dominant bands are positive; small signed residual
# contributions and noise remain possible. The MAS trace is drawn in black.

# %%
# Place the two injected populations on the candidate parameter grid.
truth = scp.zeros((shift.size, pq.size), coordset=[shift, pq])
truth[1, 1], truth[3, 3] = 0.65, 0.35
# Sum the response columns with the injected weights, then restore the 2D shape.
mqmas.data = (q_kernel.matrix @ truth.data.ravel()).reshape(mqmas.shape)
mas.data = m_kernel.matrix @ truth.data.ravel()
# Give MAS a different gain and a constant baseline for the joint fit to estimate.
mas = 1.7 * mas + 0.02 * abs(mas).max()
for seed, spectrum in zip((310, 311), (mqmas, mas), strict=True):
    spectrum += scp.normal(
        scale=0.005 * abs(spectrum).max(),
        size=spectrum.shape,
        seed=seed,
        coordset=spectrum.coordset,
    )

# Align the two experiments on one shared F2 axis, in decreasing ppm.
_ = scp.multiplot(
    [mqmas, mas],
    nrow=2,
    method=["contour", "pen"],
    sharex=True,
    height_ratios=[3, 1],
    labels=["Synthetic MQMAS", "Synthetic MAS"],
    xlim=(-16, 16),
    x_reverse=True,
    xlabel="F2 (ppm)",
    panel_kwargs=[
        dict(
            # Twelve geometric contour magnitudes per sign, from 2% to 81%.
            levels=sorted(
                sign * 0.02 * 1.4**i * float(abs(mqmas).max())
                for sign in (-1, 1)
                for i in range(12)
            ),
            cmap="viridis",
            zlim=(-float(abs(mqmas).max()), float(abs(mqmas).max())),
            positive_linestyle="-",
            negative_linestyle="--",
            linewidth=0.8,
            ylabel="Isotropic F1 (ppm)",
            equal_aspect=False,
        ),
        dict(ylabel="Intensity", color="0.2"),
    ],
    figsize=(7, 7),
)

# %% [markdown]
# ## Fit MQMAS alone and then the shared MQMAS/MAS population
#
# Both fits minimize spectral residuals under nonnegativity and the same
# second-difference penalty along the normalized parameter axes.
# `regularization=0.001` is a chosen illustrative value, not an automatically
# selected optimum. The joint fit uses `mas_weight=1` by default. Each
# experiment's residual is divided by its own peak and averaged over its
# unmasked points; doubling the point count therefore does not by itself
# double its statistical weight.
#
# The joint fit estimates a positive MAS gain and a constant MAS baseline.
# The returned gain and baseline refer to the peak-normalized inputs; they
# should not be compared directly with the injected raw-data gain 1.7.
# Predictions and residuals are returned in the original spectral units.

# %%
# Fit the same population grid, first with MQMAS alone, then with both experiments.
solo = inv.fit(mqmas, q_kernel, regularization=0.001)
joint = inv.fit(mqmas, q_kernel, mas=mas, mas_kernel=m_kernel, regularization=0.001)
# Normalize for display only; this does not impose a sum-to-one fit constraint.
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
    zlim=(0, 0.7),
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

# %% [markdown]
# The solver's population scale uses the MQMAS peak as its reference.
# For these three plots only, divide each fitted map by its sum to compare
# fractions with the injected distribution. There is no sum-to-one constraint
# in the fits. These are cell weights, not a continuous probability density.
# Line contours interpolate the coarse map for display. All three panels
# use the same cell-fraction levels: 0.02, 0.05, 0.10, 0.20, 0.35, 0.50 and
# 0.65, with a common viridis color scale from 0 to 0.70. Populations are
# nonnegative; signed spectral maps distinguish negative values by dashes.
#
# The MAS constraint can change how weights are distributed between nearby
# cells. It does not guarantee a lower MQMAS-only residual or a better
# population estimate for every choice of noise, regularization and model.
# Weakly excited or invisible MQMAS cells can be especially sensitive to the
# complementary MAS response and the model assumptions.
#
# ## Inspect residuals in both experiments
#
# `rms` and `mas_rms` are RMS errors relative to each observation's peak.
# Residual maps below share the MQMAS observation peak as their display scale;
# each panel is not separately normalized. Look for coherent residual
# structures rather than relying only on a scalar error. The same geometric
# contour levels are used in both panels: 1%, 1.4%, 1.96% and 2.744% of the
# observation peak, for each sign. The first level is twice the injected
# noise standard deviation. Smaller residuals still contribute to the RMS.
# Both panels use the same viridis scale from -0.03 to 0.03. Positive
# contours are solid and negative contours dashed; no areas are filled.

# %%
_ = scp.multiplot(
    [solo.residual / abs(mqmas).max(), joint.residual / abs(mqmas).max()],
    ncol=2,
    method="contour",
    colorbar=False,
    levels=sorted(sign * 0.01 * 1.4**i for sign in (-1, 1) for i in range(4)),
    zlim=(-0.03, 0.03),
    cmap="viridis",
    positive_linestyle="-",
    negative_linestyle="--",
    linewidth=0.8,
    labels=["MQMAS-only residual / peak", "Joint MQMAS residual / peak"],
    xlabel="F2 (ppm)",
    ylabel="Isotropic F1 (ppm)",
    figsize=(10, 4),
    equal_aspect=False,
)
_ = scp.plot_multiple(
    [mas, joint.mas_reconstructed, joint.mas_residual],
    labels=["Synthetic MAS", "Joint prediction", "Residual"],
    color=["0.2", "#b2182b", "#2166ac"],
    method="pen",
    legend="best",
    figsize=(8, 3),
)

# %% [markdown]
# ## Test the regularization and numerical choices
#
# Reuse the kernels to scan the regularization without repeating RF
# propagation. A lower residual is not sufficient to identify a unique
# distribution. Strong smoothing can spread the two injected populations
# across several cells; a visually smooth map can still give biased fractions.

# %%
print("lambda      MQMAS RMS    joint Q RMS   joint MAS RMS")
for strength in (0, 0.001, 0.01, 0.1):
    qfit = inv.fit(mqmas, q_kernel, regularization=strength)
    both = inv.fit(
        mqmas, q_kernel, mas=mas, mas_kernel=m_kernel, regularization=strength
    )
    print(f"{strength:<11g} {qfit.rms:>9.3%} {both.rms:>13.3%} {both.mas_rms:>14.3%}")

# %% [markdown]
# Before interpreting measured populations, vary the parameter range and grid
# spacing, eta assumptions, broadening, powder sampling, simulation resolution,
# starting acquisition model, spectral masks and relative MAS weight. Inspect
# accumulation at grid boundaries and stability of integrated groups of cells.
# Nonnegative weights do not require all spectral intensities to be positive:
# preserve the signed real phased data and model.
#
# Changing grids changes the discrete regularization problem. Re-evaluate
# the regularization rather than treating a fixed number as a universal
# physical smoothing scale. `normalize_columns=True` is an MQMAS-only option
# that regularizes response amplitudes; joint fits use population units and
# reject that option. The amplitude map and population map are distinct when
# RF sensitivity varies between cells.
#
# The measured PBG1 application below uses a larger acquisition-specific
# model and separate processing choices. Its runtime and scientific limits
# belong to that application; this small matched-model example does not
# qualify PBG1 site populations.

# %% [raw] raw_mimetype="text/restructuredtext"
# Continue with
# -------------
#
# * :doc:`Short synthetic workflow </examples/gallery/analysis/plot_mqmas_joint_inversion>`.
# * :doc:`API and parameter conventions </reference/mqmas_inversion>`.
#
# .. toctree::
#    :maxdepth: 1
#
#    mqmas_inversion
