PULSAR history and scope
========================

PULSAR began as a method for computing powder spectra under sample
rotation without explicit RF pulse propagation, as described in the
original paper :cite:p:`amoureux1995pulsar`. A Fortran implementation in
:cite:t:`masai` circulated mainly within the Laboratoire Catalyse et
Spectrochimie (LCS). SpectroChemPy NMR ported the method to Python and
added optimizations such as reusable powder grids, optional compiled
deposition via Numba, and parallel powder averaging for pulse sequences.

Which model to use
------------------

* **PULSAR** produces one-dimensional powder spectra for CSA and
  quadrupolar interactions in static, MAS and variable-angle settings.
  It also models finite RF preparation and detection, with up to two
  RF-active spins coupled by secular isotropic J. The coupled model
  excludes dipolar coupling. Results are SpectroChemPy datasets.
* **MRSimulator** builds spectra from spin systems and measurement methods.
  Its library covers one- and two-dimensional solid-state NMR, including
  MQVAS and STVAS. The comparison below uses its ideal central-transition
  method. See the `MRSimulator methods library
  <https://mrsimulator.readthedocs.io/en/stable/user_guide/methods_library/methods_library.html>`_.
* **SIMPSON** propagates spin dynamics with explicit pulse sequences and
  supports larger spin systems and dipolar couplings. The comparison
  below uses two spins and a scalar coupling, within PULSAR's coupled
  model. See the `SIMPSON project
  <https://inano.au.dk/about/research-centers-and-projects/nmr/software-and-tools/simpson>`_.

Benchmarks on shared models
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 31 22 21 26

   * - Simulated case
     - Reference
     - Median time, PULSAR / reference
     - Agreement
   * - ²⁷Al in Al(acac)₃, central transition at 0.5 kHz MAS
     - MRSimulator 1.0.0
     - 1,587 / 206 ms
     - Maximum bin difference 0.214% of reference peak
   * - Same model at 1 kHz MAS
     - MRSimulator 1.0.0
     - 1,387 / 190 ms
     - Maximum bin difference 0.194% of reference peak
   * - Static ¹H–¹³C doublet, J = 128 Hz
     - SIMPSON 4.2.1
     - —
     - Equal magnitudes, peak spacing 128.13 / 128.00 Hz

The MAS comparison uses one site, a field of 9.409 T, Cq = 2.923 MHz,
ηQ = 0.136, zero isotropic offset and ideal excitation at the magic angle.
Both runs use a 32 kHz window, 16,384 points and 51,681 orientations per
octant. PULSAR uses its default NumPy deposition and 29 orders. MRSimulator
uses 32 sideband slots and one γ angle. The difference is measured after
area-preserving alignment onto a shared grid and unit-area normalization,
before display broadening. Timings are medians of seven runs after
warmup with one thread on an Intel Core i5-1135G7 (Python 3.13,
SpectroChemPy 1.3.1). They include creation of the output datasets.
The alignment and plot are outside the timed region.

In the static case, SIMPSON acquires a 1,024-point FID from transverse ¹H
magnetization over 2,048 Hz. Its Fourier transform is compared with the
frequency-domain doublet from PULSAR after ideal excitation. The peak
locations agree within the 2 Hz frequency sampling and each peak carries
half the summed peak magnitude. The two programs return different
representations in this test, so runtimes are omitted.

Run either case with the :download:`benchmark script
<../../tools/benchmark_pulsar_reference.py>`. The external reference
programs are optional.

These numerical comparisons apply to the specified Hamiltonians and
settings. The `experimental Al(acac)₃ spectrum
<https://mrsimulator.readthedocs.io/en/stable/introduction/fitting_example.html>`_
was recorded at 12.5 kHz MAS. The two slow MAS spectra in :doc:`pulsar`
illustrate the ideal model.

Synthetic examples
------------------

The :doc:`simulation gallery </examples/gallery/simulation/index>` includes
static and spinning CSA, quadrupolar central transitions, a fit to a
synthetic spectrum and powder patterns with Czjzek disorder.
The :doc:`PULSAR guide <pulsar>` also simulates a spin echo.
