Czjzek distributions of quadrupolar tensors
===========================================

The NMR module can average powder spectra over an electric-field-gradient
(EFG) distribution. The standard Czjzek model describes isotropic Gaussian
disorder in the five independent coordinates of a symmetric traceless
tensor. The extended model adds that disorder to a persistent local tensor.

Construct a model and simulate its powder spectrum
--------------------------------------------------

.. code-block:: python

   from spectrochempy_nmr.czjzek import CzjzekDistribution, simulate_czjzek

   distribution = CzjzekDistribution(
       sigma_hz=0.3e6,
       cq_hz=3e6,
       eta=0.6,
       shift_hz=5000,
       shift_sigma_hz=200,
   )
   spectrum = simulate_czjzek(
       distribution,
       samples=32,
       seed=0,
       spin=2.5,
       larmor_hz=130e6,
       spinning_hz=12e3,
       spectral_width_hz=160e3,
       points=2049,
       accuracy=4,
       transitions="central",
   )

All frequency parameters use Hz. ``sigma_hz`` is the standard deviation of
one orthonormal EFG tensor coordinate expressed in quadrupolar-frequency
units. ``cq_hz`` and ``eta`` define the local tensor. Setting ``cq_hz=0``
gives the standard model. ``shift_hz`` and ``shift_sigma_hz`` describe an
independent Gaussian isotropic chemical shift. This independence is a model
assumption which should be checked for a particular material.

``distribution.sample(size=256, seed=0)`` returns weighted tensor nodes
with ``cq_hz``, ``eta``, ``shift_hz`` and ``weights`` arrays. A fixed seed
makes repeat evaluations reproducible. For quantitative fits, increase
``samples`` and the powder ``accuracy`` independently and compare the
predictions until they converge. A simulation can be sensitive to the weak
low-coupling tail of a distribution even when its central moments agree.

For ideal excitation, ``simulate_czjzek`` averages the unnormalized
component spectra and normalizes the ensemble once. Finite RF calculations
retain the simulator's ideal-reference amplitude scale and the excitation
differences between components. Pass pulse-sequence events using the
``events`` argument and use the RF coupling frequency
:math:`|\gamma| B_1/(2\pi)` for ``Pulse.rf_hz``. For half-integer spins, a
selective central-transition nutation rate can differ from that coupling
frequency. Document the experimental RF calibration before comparing
satellite intensities.

Validation scope and current uncertainties
------------------------------------------

Numerical tests check the standard distribution against its analytical
invariants, verify the zero-disorder limit against a single-tensor PULSAR
simulation, and check the averaging of finite-pulse spectra. The gallery
overlay below illustrates central line shapes. Quantitative experimental
validation remains open.

On PBG1, the extended model was constrained using the public 3QMAS map
with a finite F1 response and simulated 3Q transfer. The 1D central line
set the component weights and common width. The 20 satellite-band areas
were held out. At 32 samples per region, their summed area was 124.16%
of the central area, with 89.40% measured. The mean absolute error per
band was 1.85 percentage points.

The transfer model uses nominal RF fields and an ideal population filter.
It approximates the processed 3QMAS response and omits relaxation.
Several regional parameters are sensitive to the sampling used to fit
the map. In the 1D prediction, increasing the sample count from 16 to 32
changed the area sum by 5.89 percentage points. That variation exceeds
the 3.17-point difference between the simulated and uniform transfer
controls at 32 samples. The benchmark therefore leaves the accuracy of
satellite intensities and the inferred regional disorder uncertain.
The protocol and sensitivity checks are documented in
``src/spectrochempy_nmr/README.pulsar.md``.

Experimental gallery candidate from Dropbox Storage
---------------------------------------------------

The archive contains a useful alumina acquisition at
``/STORAGE/bruker/data/louwanda/nmr/27Al/12``. Its title identifies
``Alumine 800 C`` with 108.4 mg of sample. The acquisition records
:math:`^{27}`Al at 130.30630648 MHz, a 2.5 µs rectangular excitation pulse
and 15 W transmitter power. The processed spectrum includes tetrahedral
and octahedral regions and a sideband spacing of approximately 12 kHz.
The recorded ``MASR=5000`` and ``CNST31=12500`` differ from that spacing.
The archived files supply power and pulse duration, while the calibrated
RF coupling remains to be established.

The gallery example ``examples/simulation/plot_alumina_czjzek.py`` accepts
the local experiment directory through ``SCPY_NMR_ALUMINA_EXPERIMENT``.
For example, after retrieving an authorized local copy,

.. code-block:: sh

   SCPY_NMR_ALUMINA_EXPERIMENT="/path/to/27Al/12" \
       python examples/simulation/plot_alumina_czjzek.py

The overlay is a qualitative comparison with illustrative model
parameters. It is displayed with peak-normalized intensities. The private
experimental files are supplied locally, and the gallery also runs with
simulated components alone. The plots make it possible to inspect whether
a proposed distribution spans the observed central shapes before a
quantitative model is developed.

Additional Storage candidates
-----------------------------

Two independent cursor walks of ``STORAGE`` returned the same 52,603
files. The subsequent content pass classified all 4,433 acquisition
headers, including 123 quadrupolar records. Beyond the alumina overlay,
these candidates can support further gallery work after qualification.

* ``/STORAGE/bruker/data/aurelie/nmr/RbNO3/1`` supplies a processed
  complex 87Rb spectrum and a raw FID, with three resolved features.
  Its title gives 14.286 kHz MAS, while the status rotor fields disagree.
* ``/STORAGE/bruker/data/eddy/nmr/150113/3`` and ``/4`` supply 23Na
  single-pulse spectra on MFI samples. Their phase and offset require
  checking before fitting.
* ``/STORAGE/bruker/data/christian/nmr/E11560(nov).zip`` contains
  27Al one-pulse and MQMAS/SPAM acquisitions. Sample identification
  and processing qualification are needed to use them together.
* ``/STORAGE/bruker/data/christian/nmr/CL4627(dec).zip`` contains a
  23Na pulse-length optimization. Its peak-height maximum occurs at
  10 µs and its integral maximum at 11 µs. The calibration regime and
  sample identity remain to be established.

References
----------

* J.-B. d'Espinose de Lacaillerie, C. Frétigny and D. Massiot,
  `MAS NMR spectra of quadrupolar nuclei in disordered solids: The Czjzek model
  <https://doi.org/10.1016/j.jmr.2008.03.001>`_.
* G. Le Caër, B. Bureau and D. Massiot,
  `An extension of the Czjzek model for the distributions of electric field
  gradients in disordered solids
  <https://doi.org/10.1088/0953-8984/22/6/065402>`_.
