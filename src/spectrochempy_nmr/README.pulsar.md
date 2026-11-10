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
a static powder or magic-angle spinning at the historical rotor angle
of $54.736^\circ$. It excludes chemical shift anisotropy, spin
couplings, satellite transitions, relaxation and line broadening. MAS
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
