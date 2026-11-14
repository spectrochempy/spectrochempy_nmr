STMAS and multiplex MQMAS
=========================

STMAS correlates a satellite transition with the central transition.
MQMAS correlates a multiple-quantum coherence with the central transition.
Both workflows can use a shear to obtain an isotropic F1 dimension, but
the shear and the conversion to isotropic ppm depend on the selected
pathway. See :doc:`mqmas` for the staged processing of a conventional
Bruker States MQMAS acquisition.

The :doc:`RbNO₃ gallery example
</examples/gallery/nmr/plot_rbno3_mqmas_multiplex>` shows how to process the ⁸⁷Rb multiplex acquisition.
The multiplex acquisition requires coherence selection before either
Fourier transform. Set ``SCP_RBNO3_DATA`` to the directory containing the local copy of
experiment 206 to replay the gallery example. Keep its acquisition
parameters and ``pdata/1/procs`` alongside the ``ser``.

STMAS with States pairs
-----------------------

In experiment 15, the binary file has room for 128 traces. It contains
111 acquired traces followed by zero padding. The TopSpin processing
status specifies 110 useful traces. Use the first 55 complete pairs for
the transform, preserving the unpaired trace in the original file.

Correct the digital filter, transform and phase F2, combine the two
States branches, then apply the shear before transforming F1. For this
spin-3/2 STMAS pathway, the shear ratio is −8/9 and the F1 scale divisor
is 17/9. The example uses a shift of +0.1 ppm from the historical notebook.

Multiplex 3Q MQMAS
------------------

Experiment 206 stores 32 indirect increments, seven phase steps and two
SPAM scans per step. This third storage dimension encodes the phase
cycle. To select 3Q, project the SPAM sum and difference onto the sine
and cosine of three times the phase-step angle, then recombine them into
two States branches. Perform the frequency transforms after this
selection. The example shows the decoding explicitly because the
ordinary 2D reader does not interpret this storage layout.

The RbNO₃ gallery example uses the magnitude and a simple common F2 phase.
The PST-5 reference shows positive real RR contours after manual F2
phasing. The RbNO₃ processing leaves out historical automatic phasing, apodization and
baseline correction. A processed reference matrix is unavailable for
the RbNO₃ acquisitions.

The signed 3Q pathway for this spin-3/2 example gives an F1 scale
divisor of 34/9. With the positive FFT convention used here, the shear
coefficient is −7/9. The F1 dwell is taken from the pulse-program
increment (``IN[0]``), rather than the nominal width of the phase-cycle
storage axis.

The RbNO₃ acquisition used by the gallery example is a local input.
For the early application of 3Q MAS to rubidium nitrate, see
`Fernandez and Amoureux (1996) <https://doi.org/10.1016/0926-2040(95)01197-8>`_.
The archived acquisition used here was recorded in 2014.

.. _mqecho-pst5-rbno3-comparison:

Symmetric echo and antiecho apodization
---------------------------------------
The table compares PST-5 and the selected RbNO₃/206 3Q component
with three F2 windows. The :doc:`MQMAS tutorial <mqmas>` generates the
PST-5 contour comparison from the public acquisition.

The echo window is :math:`w_+(t_1,t_2)=\exp(-\pi L|t_2-|k|t_1|)`.
The symmetric window is :math:`[w_+(t_1,t_2)+w_+(t_1,-t_2)]/2`.
The time origin accounts for the digital group delay. We used
:math:`|k|=19/12` and :math:`L=40\,\mathrm{Hz}` for PST-5, and
:math:`|k|=7/9` and :math:`L=50\,\mathrm{Hz}` for RbNO₃/206.
These are direct-dimension settings from the Bruker processing
parameters. This variant applies the same window to both States
branches before F2 transformation, with no additional F1 window.
For RbNO₃, the 3Q selection precedes the window. The phase is held
fixed within each acquisition. PST-5 shows positive RR contours,
and RbNO₃ shows magnitude.

A and B identify the same two resonances across the three treatments within each acquisition. Widths are full
widths at half maximum (FWHM) in ppm, measured on F1 and F2 cuts
through the local 2D maxima. B/A is the ratio of their peak heights.

.. csv-table::
   :header: Acquisition, Window, "F1 FWHM A / B (ppm)", "F2 FWHM A / B (ppm)", "Height B/A"
   :widths: 16, 24, 22, 22, 16

   PST-5, None, 1.62 / 1.92, 1.89 / 2.22, 0.881
   PST-5, Echo only (40 Hz), 1.70 / 2.00, 2.06 / 2.34, 0.910
   PST-5, Symmetric (40 Hz), 1.91 / 2.24, 2.06 / 2.36, 0.925
   RbNO₃/206, None, 0.87 / 0.90, 3.30 / 3.75, 0.761
   RbNO₃/206, Echo only (50 Hz), 0.86 / 0.82, 5.55 / 4.32, 0.774
   RbNO₃/206, Symmetric (50 Hz), 0.88 / 0.85, 5.55 / 4.29, 0.778

On PST-5, symmetrization adds about 12% to both F1 widths relative
to the echo-only treatment. On RbNO₃/206, echo-only weighting already
accounts for nearly all of the F2 broadening. The F2 cut through
RbNO₃ peak A can span two nearby lobes at half height, so its FWHM
characterizes the cut as a whole.

.. note::

   Both acquisitions start after the conversion pulse. The predicted
   antiecho center lies at negative physical time for nonzero t1.
   The raw grid used for weighting begins at the negative digital group
   delay, so it still contains the mirrored center for increments
   1–13 of PST-5 and 1–8 of RbNO₃/206 (counting from zero).
   For later increments only the tail overlaps the raw grid. Overlap
   with this grid does not establish that a physical antiecho was acquired.
