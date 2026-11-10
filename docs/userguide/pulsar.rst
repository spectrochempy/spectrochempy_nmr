Simulate and fit powder spectra with PULSAR
===========================================

PULSAR is an experimental NumPy port of the Masaï solid-state NMR
simulator. It computes one-dimensional powder spectra for quadrupolar
transitions and chemical shift anisotropy (CSA). It supports static
powder, magic-angle spinning and variable-angle spinning. Finite RF
sequences can include secular isotropic J coupling.

Ideal-pulse spectra
-------------------

For a spin-5/2 central transition with two sidebands on each side,
use :func:`spectrochempy_nmr.pulsar.simulate_central_transition`.

.. code-block:: python

   from spectrochempy_nmr.pulsar import simulate_central_transition

   spectrum = simulate_central_transition(
       spin=2.5,
       cq_hz=4e6,
       eta=0.6,
       larmor_hz=100e6,
       spectral_width_hz=100e3,
       shift_hz=4500,
       spinning_hz=10e3,
       sidebands=2,
   )

The returned dataset has an increasing frequency offset axis in Hz.
Each intensity is the mass in a bin whose left edge is the coordinate.
The selected bands have unit total area. Choose a simulation window
wide enough to contain all selected bands and transitions.

Use :func:`spectrochempy_nmr.pulsar.simulate_quadrupolar` for satellites
or all transitions. This includes first- and second-order quadrupolar
effects and supports integer and half-integer spins. For CSA alone,
use :func:`spectrochempy_nmr.pulsar.simulate_csa`. Its signed anisotropy
is the principal value along z minus the isotropic shift, in Hz.
To combine CSA and quadrupolar effects, supply the CSA anisotropy,
asymmetry and optional relative rotation to either quadrupolar
function. The columns of the rotation matrix give the CSA principal
axes in the electric-field-gradient frame.

Preparation and detection
-------------------------

:func:`spectrochempy_nmr.pulsar.simulate_pulse_sequence` accepts an
ordered list of finite or ideal RF pulses, delays and coherence
selections. Use :func:`spectrochempy_nmr.pulsar.simulate_coupled_sequence`
for two RF-active spins coupled by isotropic J. Dipolar coupling is
outside the model. For example, this static spin-1/2 echo uses the
receiver phase to place the refocused signal in positive absorption.

.. code-block:: python

   from spectrochempy_nmr.pulsar import Delay, IdealPulse, simulate_pulse_sequence

   echo = simulate_pulse_sequence(
       [IdealPulse(90), Delay(200e-6), IdealPulse(180), Delay(200e-6)],
       spin=0.5,
       larmor_hz=100e6,
       spectral_width_hz=20e3,
       csa_anisotropy_hz=3000,
       csa_eta=0.5,
       receiver_phase_deg=180,
       t2_s=1e-3,
   )

Sequence output is complex and retains excitation losses. Its amplitude
scale comes from an ideal on-carrier 90° x pulse. Pass
``output="density"`` to obtain the powder-averaged deviation density
matrix at the end of preparation.

Fit an experimental spectrum
----------------------------

Use :func:`spectrochempy_nmr.pulsar_fit.fit_central_transition` on a
real, absorptive one-dimensional dataset. The coordinate can be in
frequency units or in ppm. For an absolute frequency axis, set
``origin_hz`` to the reference frequency. For ppm, provide the Larmor
frequency and use the chemical zero already encoded in the axis.

.. code-block:: python

   from spectrochempy_nmr import fit_central_transition
   from spectrochempy_nmr.pulsar_fit import CentralParameters

   result = fit_central_transition(
       experimental_dataset,
       initial=CentralParameters(3e6, 0.3, 500, 80),
       bounds=(
           CentralParameters(2e6, 0, 0, 20),
           CentralParameters(4e6, 1, 1500, 250),
       ),
       spin=2.5,
       larmor_hz=104.38e6,
       spectral_width_hz=12500,
       spinning_hz=12500,
       window_hz=(-1300, 1100),
   )

This single-site fit estimates the quadrupolar coupling, asymmetry,
shift and Gaussian width, with a nonnegative amplitude and linear
baseline. It returns the parameters, fitted spectrum and residuals on
the input axis. Start with positive coupling and constrain the
Gaussian width to at least one simulation bin. The frequency window
and parameter bounds should cover the model throughout the fit.
Repeat the calculation on a finer mesh and from another start to
check stability.

Scope of the checks
-------------------

Comparisons with historical PULSAR check the fidelity of the port.
Separate spin-Hamiltonian and time-domain calculations check selected
static and spinning cases. The experimental aluminum acetylacetonate
check covers one central band. In the experimental YAG comparison,
predicted sideband peaks agree within one measured bin. The areas
still show larger differences, so the YAG check does not establish
quantitative experimental intensity accuracy.
