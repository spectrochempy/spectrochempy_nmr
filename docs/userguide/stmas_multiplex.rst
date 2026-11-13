STMAS and multiplex MQMAS
=========================

STMAS correlates a satellite transition with the central transition.
MQMAS correlates a multiple-quantum coherence with the central transition.
Both workflows can use a shear to obtain an isotropic F1 dimension, but
the shear and the conversion to isotropic ppm depend on the selected
pathway. See :doc:`mqmas` for the staged processing of a conventional
Bruker States MQMAS acquisition.

The :doc:`RbNO₃ gallery example
</examples/gallery/nmr/plot_rbno3_mqmas_multiplex>` shows two
⁸⁷Rb acquisitions alongside the public ²⁷Al PST-5 reference.
The conventional PST-5 acquisition already stores States pairs,
while the RbNO₃ multiplex acquisition requires coherence selection
before either Fourier transform. Use local copies of the RbNO₃ experiment
directories and set ``SCP_RBNO3_DATA`` to their parent directory before
running the example. Keep the acquisition parameters and ``pdata/1``
processing parameters alongside each ``ser``.

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

The RbNO₃ maps use the magnitude and a simple common F2 phase.
The PST-5 reference shows positive real RR contours after manual F2
phasing. Each panel is normalized to its own maximum. The RbNO₃
processing leaves out historical automatic phasing, apodization and
baseline correction. A processed reference matrix is unavailable for
the RbNO₃ acquisitions.

The signed 3Q pathway for this spin-3/2 example gives an F1 scale
divisor of 34/9. With the positive FFT convention used here, the shear
coefficient is −7/9. The F1 dwell is taken from the pulse-program
increment (``IN[0]``), rather than the nominal width of the phase-cycle
storage axis.

The original RbNO₃ acquisitions are local inputs for the executable
example. The rendered figure shows the output with those inputs.
For the early application of 3Q MAS to rubidium nitrate, see
`Fernandez and Amoureux (1996) <https://doi.org/10.1016/0926-2040(95)01197-8>`_.
The archived acquisition used here was recorded in 2014.

The :ref:`mqecho-pst5-rbno3-comparison` in the gallery shows how a
symmetric exponential window changes the peak widths and relative
heights of PST-5 and RbNO₃/206. It includes the echo-only treatment
and describes the effect of the acquisition window on the antiecho.
