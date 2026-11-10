# PULSAR algorithm notes

This note describes the NumPy port in [`pulsar.py`](pulsar.py). It is a
developer reference for the PULSAR module and a source for future plugin
documentation. The implementation follows the central-transition path of
Masaï PULSAR by Christian Fernandez and Jean-Paul Amoureux. The original
sources in `masai/masai/core/fit/pulsar/f90/` use CeCILL-B.

## Model and output

The simulation models the $+1/2 \leftrightarrow -1/2$ transition of an
isolated half-integer spin $I\geq 3/2$, with second-order quadrupolar shifts, an ideal
pulse, zero evolution delay and a powder distribution. The model supports
a static powder or spinning at a specified rotor angle. The default
is the historical magic-angle setting of $54.736^\circ$. The ideal-pulse model includes optional passive isotropic J couplings. It excludes relaxation and line broadening. MAS
sidebands here are sidebands of the central transition.

```python
from spectrochempy_nmr.pulsar import simulate_central_transition

spectrum = simulate_central_transition(
    cq_hz=4.0e6,
    spin=2.5,
    eta=0.7,
    larmor_hz=100e6,
    shift_hz=4500,
    spinning_hz=10e3,
    spectral_width_hz=100e3,
    sidebands=2,
)
```

The dimensionless spin quantum number defaults to $I=3/2$.
All frequencies are in Hz. `larmor_hz` is the positive Larmor frequency
and `eta` is the quadrupolar asymmetry in $[0,1]$. The frequency origin
is the midpoint of the simulated window. A positive `shift_hz` adds
directly to the offset. A positive sideband order (n) adds
$n\nu_r$, where $\nu_r$ is the MAS rate. `sidebands=N` includes
orders $-N,\ldots,N$. A zero MAS rate selects the static model and
requires `sidebands=0`.

The `NDDataset` contains an increasing axis from $-SW/2$ to $SW/2$
with step $SW/(P-1)$ for $P$ points. Each axis value labels a **left
bin edge** in the PULSAR convention. The last point can receive mass at
the upper endpoint. The window must contain the frequency support
computed for every requested band. Each sample is integrated bin mass,
and the sum over the requested bands is one. Relative band areas are
preserved within that selection. The model does not report the fraction
of total intensity in omitted orders.

## Powder mesh

`_powder_mesh` starts with integer triples $(q-i-j,i,j)$ on a planar
triangular lattice, with $q=5\,\text{accuracy}$. It divides each
triple by its norm $r$ to project it onto the unit sphere. Radial
projection of a plane has solid-angle Jacobian proportional to
$r^{-3}$ for a fixed distance of the plane from the origin. This gives
the relative vertex weights. The common scale cancels when the spectrum
is normalized.

The central-transition model with only the quadrupolar tensor uses the
octant symmetry of the original calculation. The mesh has
$(q+1)(q+2)/2$ vertices and $q^2$ triangles. The implementation caches
the mesh by accuracy because it is independent of the spectral
parameters. The original mesh and triangle traversal are in
`edgea` and `spectrea` in `main.f90`.

## Second-order frequency harmonics

At each orientation, `_quadrupolar_harmonics` constructs the five
components $T_m$ of the rank-two tensor, for $m=-2,\ldots,2$, using
the convention in `am` in `ll.f90`. In the following expressions,
$z=\cos\beta$, $s=\sin\beta$, $c_2=\cos 2\alpha$ and
$s_2=\sin 2\alpha$. The azimuth is measured in the tensor's principal
axes and the code takes it to be zero at the pole.

$$
\begin{aligned}
T_{-2} &= \tfrac12 s^2 - \tfrac{\eta}{6}(1+z^2)c_2
                         - i\tfrac{\eta}{3}z s_2, \\
T_{-1} &= -sz(1+\tfrac{\eta}{3}c_2)-i\tfrac{\eta}{3}s s_2, \\
T_0 &= (3z^2-1-\eta s^2c_2)/\sqrt6, \\
T_1 &= -T_{-1}^*, \qquad T_2=T_{-2}^* .
\end{aligned}
$$

The Wigner coefficients $d_{mk}(\theta_r)$ rotate the tensor using
the conventions in `wigner` in `util.f90`. Only $k=\pm1,\pm2$
enter the central-transition expression. Products of rank-two
components give harmonics up to order four. For $l=0,\ldots,4$, define

$$
B_l = \sum_{k=1}^{2} a_k \sum_{m+m'=l}
             d_{mk}(\theta_r)d_{m',-k}(\theta_r)T_m T_{m'},
\qquad (a_1,a_2)=(6,3).
$$

The weights $6$ and $3$ follow from `u1s` and `u2s` in `main.f90`,
specialized to the central transition of a spin 3/2. For other
half-integer spins the ratio of the two weights stays at 2:1.
Their common spin dependence enters the frequency scale,

$$
S_I = \frac{3(I+3/2)}{4I^2(I-1/2)}, \qquad
H_0 = S_I\frac{C_Q^2}{16\nu_0}B_0 .
$$

Here $S_{3/2}=1$. For $l>0$, the code doubles the scaled $B_l$ to
include the conjugate negative harmonics. The resulting coefficients
$H_l$ are in Hz. The mean transition frequency for a vertex is
$f_0=\text{shift}+\operatorname{Re}H_0$. For a static powder,
$\theta_r=0$ and only $f_0$ is used.

## Quadrupolar satellite transitions

`simulate_quadrupolar` uses all adjacent Zeeman transitions by
default and supports both integer and half-integer spins up to 16. For
half-integer spins select `transitions="satellites"` to omit
the central transition, or `transitions="central"` to select it.
Use a window wide enough to cover the selected transitions and MAS
orders.

For the pair $m \to m-1$, define $p=m(m-1)$, $A=I(I+1)$ and
$K=3C_Q/[4I(2I-1)]$. In the physical EFG axes the first-order
frequency has the same rank-two form as a CSA tensor with
$\Delta=-K(2m-1)$ and the transverse asymmetry sign reversed.
The second-order harmonics use the Wigner products above with
transition-specific factors,

$$
a_1=-\tfrac12(24p-4A+9), \qquad
a_2=-\tfrac14(12p-4A+6).
$$

Their frequency scale is $K^2/\nu_0$ and the nonzero harmonics are
doubled as before. Both orders and any CSA contribution are combined
before computing the rotor phase for each transition. Each spectrum
is weighted by $A-p$, proportional to the squared transverse
matrix element for ideal excitation and detection. Raw contributions
are summed before normalization over the requested transitions and
orders, preserving their relative intensities. Static multiplet tests
compare the first- and second-order moments to Zeeman-basis perturbation
theory for integer and half-integer spins.

## Chemical shift anisotropy

Use `simulate_csa` for first-order CSA alone, including a spin-1/2
nucleus. This frequency model has no spin-dependent shape factor. Its
signed reduced anisotropy is $\Delta=\delta_{zz}-\delta_{\mathrm{iso}}$
in Hz, and $\eta=(\delta_{yy}-\delta_{xx})/\Delta$ lies in $[0,1]$.
The traceless principal values are
$\Delta[-(1+\eta)/2,-(1-\eta)/2,1]$. Convert a signed anisotropy in ppm
to Hz by multiplying by the positive Larmor frequency in MHz.

In the principal axes, the first-order harmonics use the same $T_l$
as above and the Wigner column $d_{l0}$,

$$
H_l = \frac{\sqrt{6}\Delta}{2} d_{l0}(\theta_r) T_l
      \begin{cases}1 & l=0\\2 & l=1,2.\end{cases}
$$

The rotor phase, bin integration and normalization are shared with
the quadrupolar calculation. The legacy CSA amplitude in `chemicalshift`
is $\Delta(3+\eta)/2$ for this convention. The tests check the static
tensor extrema and variance for both signs of $\Delta$, and compare
the MAS band areas with a Cartesian projection of the tensor along a
rotor trajectory and direct Fourier sums.

To combine CSA with the quadrupolar transition, pass `csa_anisotropy_hz`,
`csa_eta` and optionally `csa_rotation` to the central-transition
simulator. The matrix $R$ is a proper rotation. Its columns are the
CSA principal axes expressed in the electric-field-gradient (EFG)
principal frame. With no matrix, the principal frames coincide.

The code projects $D=R\,\mathrm{diag}(\delta_{ii}-\delta_\mathrm{iso})R^T$
onto the field direction along the rotor cone. Expanding the quadratic
form $\mathbf b(t)^T D\mathbf b(t)$ gives harmonics through order two.
The CSA and quadrupolar frequencies are added at each orientation
**before** integrating the rotor phase. The model uses the first-order
CSA and second-order quadrupolar secular contributions. Higher-order
mixed terms are outside this approximation.

With relative rotation, octant reflections are no longer symmetries.
Four octants cover one hemisphere. Inverting the rotor axis maps the
field trajectory to its negative with a reversal of the rotor phase
(up to a phase origin). Both frequency contributions are even in the
field direction, and the squared Fourier coefficients are unchanged
under this mapping. The test of a rotated tensor compares the combined
five-band spectrum with a whole-sphere time-domain calculation.

The relative rotation uses the physical EFG frame defined by the
Hamiltonian term $\eta_Q(I_x^2-I_y^2)$. In the legacy spherical tensor
the two transverse EFG axes are interchanged relative to this convention.
The port accounts for that interchange when CSA is present. A
quadrupolar-only powder profile is insensitive to this relabeling.

## Rotor phase and sidebands

For MAS the calculation samples one rotor cycle at angles
$\gamma_j=2\pi j/M$. Integrating the nonzero frequency harmonics
gives the phase used by the complex rotor signal,

$$
\phi_j=-\operatorname{Im}\sum_{l=1}^4
                \frac{H_l}{l\nu_r}e^{-il\gamma_j},
\qquad s_j=e^{i\phi_j}.
$$

The division by $l\nu_r$ comes from integration in time. The $2\pi$
in the phase conversion cancels the $2\pi$ in the angular rotor
frequency. The band at $f_0+n\nu_r$ has vertex weight proportional to

$$
\left|\frac{1}{M}\sum_{j=0}^{M-1}s_j e^{-in\gamma_j}\right|^2.
$$

The ideal-pulse, zero-delay restriction makes the two rotor phase sums
in `ll.f90` conjugates apart from common detection factors. The NumPy
code uses the mean for order zero and the FFT for multiple orders.
Changing the sign in the Fourier kernel or in the band shift can
interchange intensities on opposite sides of the central band.

Pass `rotor_angle_deg` to use a different rotor tilt. The angle is
measured between the rotor axis and the field. Both the zero-frequency
term and the phase modulation use the specified tilt. At zero tilt
the field direction is fixed during rotation and the sidebands vanish.

The sample count is the maximum of $q$, the rounded ratio
$SW/\nu_r$ plus one, and $2N+1$. The latter leaves distinct FFT
indices for the requested orders. The implementation limits the rotor
grid to 4096 samples, processes vertices in blocks of 128 and caps
the vertex-by-band intensity table at 64 MiB. These are computational
limits. Resolution should be checked by comparing simulations at
different mesh densities.

## Integration into frequency bins

For each band and triangle, the algorithm sorts the three vertex
frequencies as $a\le b\le c$. A frequency interpolated linearly over
a uniformly sampled triangle has a piecewise quadratic cumulative
distribution. The fraction of the triangle at or below $f$ is

$$
F(f)=
\begin{cases}
0, & f<a, \\
(f-a)^2 / [(b-a)(c-a)], & a\le f\le b, \\
1-(c-f)^2 / [(c-b)(c-a)], & b<f<c, \\
1, & f\ge c .
\end{cases}
$$

This expression assumes distinct vertex frequencies. When $a=b$ or $b=c$,
the surviving branch is used for interior edges. If all three
frequencies land in the same bin, that bin receives the full triangle
mass. The mass is the sum of the three vertex weights, including rotor
intensity when applicable. Taking differences of $F$ across bin edges
integrates the triangle's contribution without assigning it to a single
sample. This is the purpose of the original `tent` routine.

The NumPy implementation flattens the bin spans into batches with a
target of 262,144 triangle-bin pairs. It resets the CDF at the first bin
of every triangle and uses `np.bincount` to add contributions to common
bins. This avoids a Python loop per triangle while keeping temporary
storage proportional to a batch, or to the span of one triangle if it
exceeds the batch target. Only the final summed spectrum is normalized.

## Powder after a finite RF pulse

`simulate_rectangular_pulse` propagates the thermal deviation
density matrix $\rho_0=I_z$ through a rectangular pulse for a static or spinning
powder. Use a positive pulse duration in seconds, RF nutation frequency
in Hz and a phase in degrees. The transmitter is on the simulation
carrier. The model retains the secular quadrupolar and CSA terms during
the pulse, so a shift relative to the carrier detunes the excitation.

The adjacent transition offsets determine the diagonal level energies
up to a constant. In descending magnetic quantum-number order, the
difference $E_{m-1}-E_m$ is the transition offset. For each orientation
the RF term is $\nu_1(I_x\cos\phi+I_y\sin\phi)$. The total
Hamiltonian in Hz is constant during a static pulse and is diagonalized
with a Hermitian eigensolver. The propagator is
$U=\exp(-2\pi i H t)$ and the final state is $U\rho_0 U^\dagger$.

The detected weights are the adjacent coherences multiplied by their
transverse matrix elements. Real and imaginary weights are deposited
separately on the same frequency grid. The receiver convention maps an
ideal on-carrier $90^\circ$ x pulse to positive absorption. This ideal
pulse defines the amplitude scale for the complex spectrum, so finite
pulse excitation losses and phase shifts are retained. Tests compare
the on- and off-resonance spin-1/2 response with a Bloch rotation and
check the hard-pulse limit for a quadrupolar powder.

Under MAS the simulator samples starting rotor phases uniformly.
Finite preparation can distinguish octant reflections even for aligned
tensors, so it uses at least four octants. It uses all eight for a
rotated CSA tensor because a time-ordered RF pulse can distinguish
the trajectories of antipodal rotor axes. The RF rotor grid is rounded
up to an even count so a half-turn maps sampled phases onto themselves.
Each
pulse step evaluates the offset harmonics at its midpoint, then applies
the constant-Hamiltonian propagator. The step is at most
`max_rf_step_s` or 1/32 of a rotor period, whichever is smaller.
The time-step test checks convergence as this step is reduced.

For acquisition, let $a(\gamma)$ be the complex detected coherence at
the final rotor angle and $s(\gamma)$ the periodic phase factor from
the offset harmonics. Band $n$ is the product of two Fourier averages,

$$
\langle s(\gamma)e^{-in\gamma}\rangle\,
\langle a(\gamma)s(\gamma)^*e^{in\gamma}\rangle .
$$

The averages account for detection during rotor modulation and for the
phase-dependent state prepared by the pulse. For a hard pulse $a$ is
constant and the product reduces to the squared coefficient of the
ideal-pulse model. The RF spectrum retains intensity losses into
omitted orders. The implementation batches density matrices and limits
persistent complex work arrays to 64 MiB and propagation to 4096 steps
per pulse. A batch of rotor phases must also fit within 131,072 matrix
elements for at least one powder vertex.

## Pulse sequences and free evolution

`simulate_pulse_sequence` takes an ordered list of immutable events.
`Pulse(duration_s, rf_hz, phase_deg, offset_hz)` is rectangular;
`IdealPulse(flip_deg, phase_deg)` is an instantaneous rotation;
`Delay(duration_s)` is free evolution; and
`Coherence(order)` projects the density matrix onto a chosen
row-minus-column magnetic quantum-number difference. Acquisition
detects order +1 in this convention.

During a delay the secular Hamiltonian is diagonal at all times.
Its diagonal entries commute, so the rotor harmonics integrate
analytically. The propagator uses the sum of the mean offset times
the delay and the difference of the periodic phase at both endpoints.
Each RF pulse's phase is defined at its onset. An offset transmitter
uses its own rotating frame during that pulse, then transforms the
density matrix back to the simulation carrier.

Optional `t2_s` multiplies off-diagonal elements by
$\exp(-t/T_2)$ after each timed event, the eventwise approximation
used here for transverse relaxation. The receiver applies
$\exp(i\phi_\mathrm{rec})$ to the detected signal. RF always propagates
all levels. The `transitions` option restricts acquisition to the
central transition, the satellites, or all transitions, and the
selected transitions set the reference amplitude for an ideal x90
pulse. An equal-delay Hahn echo tests refocusing in a static powder
and over complete MAS rotor periods, including the predicted
transverse attenuation.

## Combining preparation pathways

Pass a `PhaseCycle(scans, coefficients=None)` as the sequence input
to combine several explicit pulse or coherence-selection pathways.
Each scan is an independent sequence starting from thermal equilibrium.
With no coefficients, the simulator takes their arithmetic mean,
as in the historical coherence-pathway list. Explicit real or complex
coefficients are used as given and can encode receiver phase cycling.
For a spectrum, each scan is detected at its own final rotor phase
before the signals are combined. For density output, complex weighting
returns the linear combination of prepared deviation matrices.

## Isotropic J coupling to passive partners

`j_partners` accepts a sequence of `PassiveJ(spin, j_hz)` objects.
For a partner that is never pulsed, the secular isotropic term
$J I_z S_z$ is diagonal in its magnetic quantum number $m_S$.
The observed-spin transition therefore gains the offset $Jm_S$.
For multiple partners, sum their offsets. The ensemble uses equal
populations of the partner states. Equivalent offsets are combined
with their multiplicities, and the state product is limited to 256.

For an ideal pulse, the offsets translate the bands without changing
their modulation. For a finite preparation sequence the same offsets
also act during pulses and delays. The simulator therefore averages
the complex responses after propagating each distinct offset. Tests
check the 1:2:1 triplet for two equivalent spin-1/2 partners and the
opposite phase evolution of an isotropic doublet.

## Two RF-active spins with isotropic J coupling

`simulate_coupled_sequence` propagates the product density matrix of
two `SpinSite` instances. Each site has its own spin, Larmor
frequency, shift, quadrupolar and CSA tensors, and optional transverse
relaxation time. An EFG rotation maps the site into the common sample
frame. Its CSA rotation maps its CSA axes into that site's EFG frame.
When rotating into the common rotor frame, the code transforms the
phase origin of each nonzero harmonic as well as the rotor direction.

The product Hamiltonian includes the diagonal term $J I_z S_z$.
The initial thermal deviation matrix is proportional to
$I_z+(\nu_S/\nu_I)S_z$. A `Pulse` can apply both RF channels
simultaneously using the `partner_rf_hz`, `partner_phase_deg`
and `partner_offset_hz` fields. `IdealPulse` accepts an
independent partner flip and phase. `Delay(decouple_j=True)`
removes the active coupling for that delay. Coherence selection acts
on the observed spin and retains the unaffected partner marginal.

Acquisition detects observed-spin coherences at fixed partner
projection. With this Hamiltonian and the descending magnetic
basis, the observed offset is $\nu_{m\rightarrow m-1}-Jm_S$.
The result averages the partner states and can include additional
passive partners coupled to the observed spin. The manifold limit is
64 product levels, with further work-array limits in MAS. Tests
recover the passive limit and check refocusing by a partner π pulse
or a decoupled delay. Dipolar coupling and nonsecular terms are outside
this model.

## Density matrix after preparation

`output="density"` on either sequence function returns the
complex powder-averaged deviation density matrix. This evaluates the
prepared state without an acquisition window, receiver phase, or
spectral normalization. In spinning simulations the spectral width
still controls the number of rotor-phase samples. Sideband selection
applies only to spectra.

The average uses the same triangle vertex multiplicities and
solid-angle weights as the spectrum, and averages the rotor phases
and passive partner states. The product basis has the partner
index varying fastest, with each spin in descending magnetic order.
Reshape the two-spin matrix to (observed, partner, observed, partner)
to obtain the reduced matrices by partial trace. Divide a partial
trace by the traced spin's dimension to use the same deviation
scale as the single-spin thermal initialization.

## Receiver bandwidth

Optional `probe_bandwidth_hz` applies the legacy Lorentzian
receiver weighting at each vertex and band frequency before depositing
its intensity. For offset $f$ from the simulation carrier and full
width at half height $W$, the factor is $1/[1+(2f/W)^2]$. The original
quality-factor parameter corresponds to $W=\nu_0/Q$ for $Q>0$.
Leaving the bandwidth unset gives a flat receiver response.

The ideal powder entry points retain unit area over the requested
transitions and bands after this weighting. Pulse-sequence spectra
retain the loss in absolute signal relative to the ideal x90 reference.
The receiver weighting does not affect the prepared density matrix.

## Verification and performance

The [tests](../../tests/test_pulsar.py) separate three kinds of checks.

- **Numerical regression against historical PULSAR.** Three axial MAS
  profiles in Masaï's `res.npy` check the peak, centroid and second
  moment. A nonaxial case at 10 kHz MAS with orders $-2$ through $+2$
  checks asymmetric areas against a rebuilt Fortran reference. These
  comparisons establish agreement for the selected cases. The historical
  implementation can itself contain errors.
- **Checks independent of the Fortran executable.** The static axial
  centroid is compared with $-C_Q^2/(40\nu_0)$, obtained by angular
  averaging the analytic spin-3/2 frequency. A nonaxial static test
  builds the Zeeman and quadrupolar Hamiltonians as spin matrices in the
  principal axes. It computes the second-order shifts from off-diagonal
  matrix elements in the Zeeman eigenbasis, then averages over the
  whole sphere using Gauss–Legendre quadrature in the polar cosine and
  a uniform azimuth grid. The test compares the centroid and standard
  deviation with the binned spectrum for spins 3/2, 5/2 and 7/2.
  For the MAS check, the field
  sweeps a cone around each sampled rotor axis. The same spin matrices
  give the instantaneous shift. Trapezoidal integration of the shift
  after subtracting its cycle average gives a time-domain phase.
  Direct Fourier sums and a whole-sphere average give the relative
  areas for five resolved bands for spins 3/2 and 5/2, independently of
  the port's Wigner
  harmonics, FFT and triangular mesh. The zero-coupling case checks
  that MAS sidebands vanish. Experimental spectra are still needed
  to assess the model against measurements.
- **Numerical edge cases.** Tests cover degenerate triangle frequencies,
  the upper bin edge, integer scalar types and the memory limit.

On the development machine, batching reduced the 8192-point,
`accuracy=16` central-band simulation from about 125 ms to 17 ms.
An illustrative adjustment on a synthetic noisy MAS profile took
0.47 s after batching, down from 4.13 s, with 32 evaluations in both
runs. The timings are for the central band. With several bands the
FFT supplies their intensities together, while powder integration
is repeated per band.

The [sideband fit benchmark](../../tools/benchmark_pulsar_sideband_fit.py)
fits the central band and two sidebands on each side at 8 kHz MAS.
It estimates the quadrupolar coupling, asymmetry and shift from a
synthetic spectrum generated at accuracy 32 with Gaussian broadening
and noise at 0.2% of the peak. The fit uses the same known broadening.
Run it with the plugin installed and `OPENBLAS_NUM_THREADS=1` for a
single-threaded comparison. On the development machine, accuracy 8 took
0.42 s (32 evaluations) and recovered the shift to within 13 Hz.
Accuracy 16 took 1.34 s (36 evaluations) and recovered it to within 3 Hz.
This benchmark measures fit cost and sensitivity to the powder mesh.
Experimental spectra are still needed to assess model accuracy.

## Fitting an NDDataset

The plugin exposes `fit_central_transition` for a real, phased 1D
spectrum. For an axis in frequency units, pass the reference as
`origin_hz` if the coordinates are absolute or use another origin.
A ppm coordinate already defines chemical zero. PULSAR converts
ppm to offsets using the positive Larmor frequency and preserves
the original axis and mask on the fitted spectrum and residuals.

```python
from spectrochempy_nmr import fit_central_transition
from spectrochempy_nmr.pulsar_fit import CentralParameters

result = fit_central_transition(
    experimental_dataset,
    initial=CentralParameters(3.0e6, 0.3, 500, 80),
    bounds=(
        CentralParameters(2.0e6, 0.0, 0, 20),
        CentralParameters(4.0e6, 1.0, 1500, 250),
    ),
    larmor_hz=104.38e6,
    spectral_width_hz=12500,
    spinning_hz=12500,
    spin=2.5,
    window_hz=(-1300, 1100),
)
```

The optimizer fits `Cq`, asymmetry, shift and Gaussian FWHM. Any
specified CSA is fixed. For each candidate, it projects a nonnegative
scale and linear baseline by least squares. The fit excludes masked
or nonfinite observations and reports the RMS in the dataset's
intensity units and the number of residual evaluations. The model
window must cover every selected band over the parameter bounds.
Use separate starts and mesh refinements when judging parameter
stability.

## Experimental check with aluminum acetylacetonate

The [MRSimulator fitting tutorial](https://mrsimulator.readthedocs.io/en/stable/introduction/fitting_example.html)
provides a Bruker whole-echo acquisition of aluminum acetylacetonate
at 9.4 T. Use its supplied, phased `pdata/1/1r` for this check.
Download and extract the
[original archive](https://ssnmr.org/sites/default/files/mrsimulator/Al_acac3_0.zip)
locally. The archive used here has SHA-256
`d0c32c470efe29d4e3236812b0064ddf558a888d8e8f0d93a8d4430085358b60`.
The repository contains the [qualification script](../../tools/qualify_pulsar_alacac.py)
and does not redistribute the acquisition.

```sh
OPENBLAS_NUM_THREADS=1 python tools/qualify_pulsar_alacac.py /path/to/pdata/1
```

The script reads the processed real spectrum with nmrglue-ng and uses
`OFFSET`, `SW_p`, `SF` and `SI` to construct its frequency axis. It
uses every sixteenth point and fits the window from −1300 to +1100 Hz
relative to the processed carrier. The model is a spin-5/2 central
transition at a single site. The fit estimates the coupling, asymmetry,
offset and Gaussian width, and solves for an amplitude and a linear
baseline at each evaluation. It uses 12.5 kHz MAS as stated in the
tutorial and in the spectrum title. The acquisition's `MASR` field
instead reads 4.2 kHz. Rerunning with `--rotor-hz 4200` at accuracy 16
changes the fitted coupling by about 0.10 kHz and the residual RMS
from 1.94% to 1.96% of the peak. The central-band fit cannot establish
which rotor rate was used.

At accuracy 16, the fit gave **2.923 MHz**, **0.140** asymmetry,
**+475.5 Hz** offset and **65.1 Hz** Gaussian FWHM. The fit took about
0.86 s for 102 residual evaluations on the development machine.
Doubling accuracy to 32 changed the coupling by 0.81 kHz and left the
residual RMS at 1.94% to two decimal places. The tutorial reports an
asymmetry of 0.136 and a Gaussian FWHM of 63 Hz, using its own processing
and a model of echo truncation. The PULSAR fit does not model that
truncation explicitly. The offset above uses the processed carrier
rather than the tutorial's external chemical-shift reference. This
check covers one measured central band. It does not qualify
experimental sideband intensities.

## Experimental check of YAG satellite bands

The [MRSimulator YAG example](https://mrsimulator.readthedocs.io/en/latest/fitting/1D_fitting/plot_4_27Al_YAG.html)
provides a processed $^{27}$Al MAS spectrum from DMFIT with two aluminum
sites. Download the
[original CSDF](https://ssnmr.org/sites/default/files/mrsimulator/27Al%20Quad%20MAS%20YAG%20400MHz.csdf)
locally. The file used here has SHA-256
`707eb4876c4b843ad86a25d41075d35d56318de45b1aa2cb98995821e5804f6a`.

```sh
OPENBLAS_NUM_THREADS=1 python tools/qualify_pulsar_yag.py "/path/to/yag.csdf"
```

The [script](../../tools/qualify_pulsar_yag.py) uses the two fitted
tensors, MAS rate and Gaussian width from the MRSimulator example.
Those parameters were obtained from this same experiment. PULSAR
simulates all spin-5/2 transitions on a wide axis and interpolates
them onto the measured grid. It subtracts the baseline estimated in
the example's noise window and calibrates the two site amplitudes
using only data within 10 kHz of the frequency origin. The subsequent
comparison covers 16 windows of width 6 kHz centered on MAS orders
−8 to −1 and +1 to +8.

At accuracy 8 with 96 sideband orders on each side, the pointwise
correlation in those windows is 0.97745. The median peak difference
is zero and the maximum is one measured bin (122 Hz). The median
relative area error is 13.70%, with a maximum of 35.54%. Increasing
accuracy to 16 and the order count to 120 yields a correlation of
0.97747 and nearly the same area errors. The positions are reproduced
more closely than the intensities. This comparison does not isolate
which effects of excitation, processing or model assumptions explain
the residual intensity differences.

## Aluminum MAS dataset with documented RF excitation

The [Bernasconi et al. archive](https://doi.org/10.5281/zenodo.17130697)
includes the Bruker experiment `raw data/PBG1_2,5mm/1` with its FID, phase
cycle, acquisition parameters and processed spectrum. The `raw data.rar`
used here has SHA-256
`9dea155c54c70a3c0ea0ee13ccedda0c36ba286d7962b60795c8e9dbdbcce405`.
The precompiled `pf_zg` sequence applies one rectangular `p1` pulse before
direct acquisition. The experiment title specifies a 50 kHz RF field at
32 W. The acquisition record gives a 0.5 µs pulse, 195.439811 MHz carrier,
30 kHz MAS, 2.5 MHz acquisition width, 0.1 s recycle delay and nominal 6.5 µs
receiver dead time. The nominal free-spin flip angle is 9°. For simulations
on the processed chemical-shift reference, the carrier offset is about
+4.409 kHz, obtained from `SFO1 - SF`. The acquisition value `O1 = 340 Hz`
is relative to the instrument's base frequency `BF1`.

```sh
python tools/measure_pulsar_pbg_bands.py "/path/to/raw data/PBG1_2,5mm/1"
```

The [measurement script](../../tools/measure_pulsar_pbg_bands.py) uses
the real TopSpin spectrum on its published frequency grid (152.59 Hz per
point). It subtracts a separate straight baseline for each order using
the flanks 8.5–12.5 kHz from the nominal band center. It integrates within
±6 kHz of each center and repeats the measurement at ±4.5 and ±7.5 kHz to
expose sensitivity to the choice of window. Each area is reported as a
percentage of the central **window** area at the same width. Across the
20 satellite windows (orders ±1 to ±10), the ±6 kHz areas range from
2.26% to 7.86% of the central window. Their sum varies from 84.64% to
91.06% of the central window across the three widths. An independent check
using shoulder medians instead of a linear baseline changed individual
±6 kHz ratios by up to 0.67 percentage points.

These measurements supply experimental targets for a finite-pulse
comparison. The sample is an amorphous binder with multiple aluminum
environments, as described in the [associated article](https://doi.org/10.1016/j.jnoncrysol.2026.124054).
A quantitative test of PULSAR still requires a sample model determined
independently of these windows and an assessment of the receiver and
processing response. The TopSpin spectrum already has a baseline
correction. The window ratios describe that processed spectrum and
include any contribution from overlapping transitions.

## Experimental check of phosphorus CSA bands

The [MRSimulator phosphorus example](https://mrsimulator.readthedocs.io/en/v1.0/fitting/1D_fitting/plot_1_31P_Na2PO4_MAS.html)
uses a DMFIT $^{31}$P MAS spectrum labeled sodium phenylphosphate in
the file metadata. Download the
[CSDF](https://ssnmr.org/sites/default/files/mrsimulator/31P%20Phosphate%206kHz.csdf)
locally. Its SHA-256 is
`78dcf44f103f48e63ffa79af674dbce072cc5ca6890dc5279a7504e4440024c8`.

```sh
OPENBLAS_NUM_THREADS=1 python tools/qualify_pulsar_phosphate.py "/path/to/phosphate.csdf"
```

The [script](../../tools/qualify_pulsar_phosphate.py) uses the published
isotropic shift, shielding anisotropy, asymmetry and rotor rate from
the MRSimulator fit to this same spectrum. It changes the sign of the
shielding anisotropy to obtain the PULSAR chemical-shift anisotropy.
It subtracts the baseline estimated below −100 ppm and integrates
the measured spectrum and simulation over seven windows of half-width
1.4 kHz around the corresponding band positions. Each set of areas
is normalized over those windows. No intensities are fitted.

The maximum absolute difference in the resulting area fractions is
0.98 percentage point, with a sum of absolute differences of 2.22
percentage points. With the sign reversed these become 11.89 and
34.00 percentage points. This is a check of the CSA sign convention
and the area distribution for one experimental spectrum.
