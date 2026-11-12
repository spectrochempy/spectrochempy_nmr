Processing conventions
======================

Development checkout
--------------------

.. code-block:: console

   python -m pip install -e .

Build records
-------------

``docs/compatibility.toml``, ``docs/build-manifest.txt`` and
``docs/phase2b-validation.md`` are historical phase-2b build records. They do
not describe the current development dependencies. The active RTD requirements
are in ``docs/requirements-rtd.txt``.

Agilent quadrature detection
----------------------------

The default ``quadrature="auto"`` recognizes versioned pulse-program
sources. Pass the exact source used for acquisition with
``pulse_program_source="path/to/NOESY.c"`` or include the source next to
``fid`` and ``procpar``, named after ``seqfil`` with a ``.c`` suffix.
The initial catalogue recognizes the audited OpenVnmrJ `NOESY.c
<https://github.com/OpenVnmrJ/OpenVnmrJ/blob/0f6b0a4664def0cdf2ee9eb37cb2df7369f38ca3/src/psglib/NOESY.c>`_
as States-TPPI when the name and SHA-256 match and ``d2=0`` establishes
the first increment. The source must correspond
to the acquisition. A sequence name or a two-branch phase array alone
cannot distinguish States from States-TPPI.

The reader logs the choice at INFO level and records the mode and its
source in ``fid.meta.acquisition["quadrature_selection"]``. For an unknown
program or absent source, automatic selection leaves the mode unset and
F1 processing guarded. Reading the stored traces and the historical F2
operation remain available. Quadrature selection requires an nmrglue-ng
provider that supports it and a SpectroChemPy core with FFT
prevalidation (1.2.0 or later).

TopSpin processing metadata
---------------------------

When a TopSpin read includes a ``procs`` file, the dataset can expose its
processing parameters in ``dataset.meta.nmr_processing["vendor_profile"]``.
The profile records the vendor, the source file, and the parameter values.
Replay requires a supported combination of acquisition and processing
settings, as described in :ref:`topspin-xfb-replay`.

For an ``Experiment.process()`` result, the processing trace in
``result.meta.nmr_processing["scp_processing"]`` distinguishes the
``requested`` arguments from the ``applied`` operations.

The core's ``FFT_PREVALIDATION_API >= 1`` hook validates the selected
axis before permutation, resizing or mutation. Native channel workflows
and profiles with undetermined indirect encoding require this hook.

Phase estimation from receiver timing
-------------------------------------

For a spectrum produced with ``fftshift(fft(fid))`` on an increasing
frequency grid, use the measured receiver timing to fix the slope and fit
only the constant phase. A rectangular hard 90° pulse contributes
approximately twice its duration divided by pi to the effective delay.
Use the time from the pulse end to the first complex sample and account for
any digital filter correction separately.

.. code-block:: python

   from nmrglue.process.proc_autophase import phase_slope_from_delay
   from spectrochempy_nmr import autophase

   slope = phase_slope_from_delay(10_000, 8e-6, hard_90_pulse_s=10e-6)
   result = autophase(spectrum, fixed_p1=slope)

.. _mqmas-qualification:

MQMAS conventions and qualification
-----------------------------------

XFSHEAR uses the equivalent of TopSpin's XHT2. It extends the
real F2 spectrum with zeros to twice the F2 FFT size for the Hilbert
transform. That extension moves the periodic wraparound farther from
the retained bins during reconstruction of the imaginary component.

``xf2()`` uses the core FFT along ``x`` with the TopSpin XF2 convention.
On an uncorrected Bruker States dataset, the call
``dataset.fft(dim="x", size=4096, convention="topspin-xf2")`` computes
the positive-exponent FFT on the TopSpin F2 grid. The MQMAS helper also
prepares the acquired samples and records the state needed by XFSHEAR.
With ``correct_digital_filter=True``, ``xf2()`` removes the phase ramp
caused by the digital filter on the F2 frequency grid. The
``correct_digital_filter`` option controls this operation.

For positive odd coherence order :math:`p` and nuclear spin :math:`I`,
the signed slope and isotropic F1 scale divisor are

.. math::

   R = \frac{p[-17p^2 + 36I(I+1)-10]}{9[4I(I+1)-3]},
   \qquad D = p-R.

For spin 5/2 and triple-quantum coherence, :math:`R=19/12` and
:math:`D=17/12`.

The TopSpin comparison with PKNL disabled uses
``correct_digital_filter=False``. The gallery uses automatic correction.
TopSpin reconstructs the imaginary F2 component on a grid twice the F2
FFT size. With this recipe, its inverse and final forward F2 transforms
cancel algebraically. The four output planes have been compared with
TopSpin 4.5.0 at F2 SI 2048 and 4096, including a case with nonzero F2
phase and F1 shift. An isolated RR point at a tile boundary in the
TopSpin output dominates the residual in the TopSpin comparison.
