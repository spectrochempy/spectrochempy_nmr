# PULSAR algorithm notes

This note describes the NumPy port in [`pulsar.py`](pulsar.py). It is a
developer reference for the PULSAR module and a source for future plugin
documentation. The implementation follows the central-transition path of
Masaï PULSAR by Christian Fernandez and Jean-Paul Amoureux. The original
sources in `masai/masai/core/fit/pulsar/f90/` use CeCILL-B.

## Model and output

The simulation models the $+1/2 \leftrightarrow -1/2$ transition of an
isolated spin $I=3/2$, with second-order quadrupolar shifts, an ideal
pulse, zero evolution delay and a powder distribution. The model supports
a static powder or magic-angle spinning at the historical rotor angle
of $54.736^\circ$. It excludes chemical shift anisotropy, spin
couplings, satellite transitions, relaxation and line broadening. MAS
sidebands here are sidebands of the central transition.

```python
from spectrochempy_nmr.pulsar import simulate_central_transition

spectrum = simulate_central_transition(
    cq_hz=4.0e6,
    eta=0.7,
    larmor_hz=100e6,
    shift_hz=4500,
    spinning_hz=10e3,
    spectral_width_hz=100e3,
    sidebands=2,
)
```

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
enter the spin-3/2 central-transition expression. Products of rank-two
components give harmonics up to order four. For $l=0,\ldots,4$, define

$$
B_l = \sum_{k=1}^{2} a_k \sum_{m+m'=l}
             d_{mk}(\theta_r)d_{m',-k}(\theta_r)T_m T_{m'},
\qquad (a_1,a_2)=(6,3).
$$

The weights (6) and (3) follow from `u1s` and `u2s` in
`main.f90`, specialized to the central transition. The code multiplies
$B_l$ by $C_Q^2/(16\nu_0)$ and doubles $l>0$ to include the
conjugate negative harmonics. Call the resulting coefficients $H_l$
in Hz. The mean transition frequency for a vertex is
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
  averaging the analytic spin-3/2 frequency. The zero-coupling case
  checks that the MAS sidebands vanish. Broader physical validation
  requires other independent results and experimental spectra.
- **Numerical edge cases.** Tests cover degenerate triangle frequencies,
  the upper bin edge, integer scalar types and the memory limit.

On the development machine, batching reduced the 8192-point,
`accuracy=16` central-band simulation from about 125 ms to 17 ms.
An illustrative adjustment on a synthetic noisy MAS profile took
0.47 s after batching, down from 4.13 s, with 32 evaluations in both
runs. The timings are for the central band. With several bands the
FFT supplies their intensities together, while powder integration
is repeated per band. Performance for experimental adjustments
with sidebands remains to be measured.
