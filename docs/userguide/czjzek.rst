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

The PBG1 validation notes in ``src/spectrochempy_nmr/README.pulsar.md``
document the separate public experiment and its held-out satellite areas.
The empirical 3QMAS ridge discretization used there shows strong
sensitivity to the low-coupling tail. A Czjzek distribution supplies a
physical tensor model for that tail, with parameters requiring their own
experimental constraints.

References
----------

* J.-B. d'Espinose de Lacaillerie, C. Frétigny and D. Massiot,
  `MAS NMR spectra of quadrupolar nuclei in disordered solids: The Czjzek model
  <https://doi.org/10.1016/j.jmr.2008.03.001>`_.
* G. Le Caër, B. Bureau and D. Massiot,
  `An extension of the Czjzek model for the distributions of electric field
  gradients in disordered solids
  <https://doi.org/10.1088/0953-8984/22/6/065402>`_.
