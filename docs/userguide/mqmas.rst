Processing MQMAS spectra
========================

MQMAS separates isotropic and second-order quadrupolar contributions for
half-integer quadrupolar nuclei. An MQMAS spectrum generally contains tilted
ridges. Shearing removes their quadrupolar slope, and scaling the indirect
axis gives an isotropic chemical-shift axis.

Shearing a frequency-domain map
-------------------------------

Start with a reconstructed, Fourier-transformed spectrum whose physical F2
and F1 coordinates are ``x`` and ``y``. Complete digital-filter correction
and phasing before shearing. Both grids must be uniform, unmasked and finite,
in Hz or in ppm with frequency references. Confirm the reference of each
axis: an acquisition carrier or an already scaled isotropic axis is not
interchangeable with the unsheared MQ frequency axis.

For a spin-5/2 nucleus observed through its triple-quantum coherence:

.. code-block:: python

   experiment = scp.nmr.Experiment(spectrum)
   isotropic = experiment.shear_mqmas(spin=2.5, coherence_order=3)

The method returns a new ``NDDataset``. It preserves the source, signal units
and component labels, records its parameters in ``meta.mqmas_shearing``, and
adds a history entry. Transposition preserves the meaning of ``x`` and ``y``.
Real maps, four real ``rr/ri/ir/ii`` planes, and the plugin's two complex
``S_R/S_I`` spectral planes are supported. Shear all available components
before selecting a component for display.

Conventions
-----------

For a positive odd coherence order :math:`p` and nuclear spin :math:`I`,
the implemented signed quadrupolar ridge slope is

.. math::

   R = \frac{p[-17p^2 + 36I(I+1)-10]}{9[4I(I+1)-3]}.

The coordinate transformation is

.. math::

   f_{\mathrm{iso}} =
   \frac{f_{\mathrm{MQ}} - R(f_{\mathrm{SQ}}-f_{\mathrm{pivot}})}{p-R}.

For 3Q of a spin-5/2 nucleus, :math:`R=19/12` and :math:`p-R=17/12`.
For 3Q of a spin-3/2 nucleus, :math:`R=-7/9` and :math:`p-R=34/9`.
The method assumes the input uses this sign convention for the MQ frequency;
confirm it when importing results from another processing package.

The default pivot is zero Hz on the calibrated F2 chemical-shift frequency
axis. A different pivot translates the final isotropic axis relative to this
convention; it is not a phase-correction pivot. Pass a frequency explicitly:

.. code-block:: python

   isotropic = experiment.shear_mqmas(
       spin=2.5, coherence_order=3, pivot=2500 * scp.ur.Hz
   )

The output F1 frequencies are divided by :math:`p-R`. Conversion of this
isotropic axis to ppm uses the F2 reference frequency. The F2 axis is unchanged.
Sample amplitudes receive no additional normalization; integration in the
rescaled F1 coordinate therefore includes the coordinate scale change.

Limits and qualification
------------------------

Fourier interpolation is periodic in F1: signal shifted past an edge folds
back at the opposite edge. Inspect the full spectral width and any aliasing
before interpreting a cropped region. This operation does not increase
resolution or remove spinning sidebands. A recorded shearing operation cannot
be applied twice to the same result.

Replaying the rectangular TopSpin recipe
----------------------------------------

Use a normalized Bruker States SER with both corrections disabled. The FFT sizes
are given in physical complex points in F1, F2 order. For MQMAS/51 the
comparison used all 64 F1 increments and the first 2048 complex samples in F2.

.. code-block:: python

   raw = scp.nmr.read_topspin(
       "MQMAS/51/ser", remove_digital_filter=False, remove_dc_offset=False
   )
   spectrum = scp.nmr.Experiment(raw).replay_xfshear(
       spin=2.5, size=(512, 2048)
   )

The qualified recipe corresponds to ``xfshear noabs 0`` with rectangular
windows, phase and baseline correction disabled, and PKNL off. The first
F2 FFT supplies real traces. The two States branches are rotated as a
function of F2 frequency and F1 time. The Hilbert transform reconstructs
F2 quadrature on a grid twice the F2 FFT size. With these settings the
inverse and final forward F2 transforms cancel, so the reconstructed
branches pass directly to the F1 FFT. The output uses the direct processing
reference and applies the MQMAS scale in F1.

On MQMAS/51, both axes agree within 5e-9 ppm. With one common amplitude
factor for all planes, the relative L2 errors against TopSpin 4.5.0 are
5.94e-4 for RR, 5.78e-7 for RI, 5.10e-7 for IR and 4.97e-7 for II.
One RR point at the F1 edge dominates its residual. The same point
appears after TopSpin XFB on an unrotated XIF2 snapshot. On the tested
tiled grids its F2 index was SI/2 minus XDIM (zero-based). The plugin retains
the input reader's amplitude normalization.

TopSpin warns when the first F2 FFT has fewer than twice as many bins
as retained complex samples. To meet that recommendation with the same
2048 retained points, use ``size=(512, 4096), direct_points=2048``.
The warning disappears at that size. The isolated RR discrepancy
persists at the corresponding tile boundary.

The :doc:`PST-5 gallery notebook </examples/gallery/nmr/plot_replay_mqmas_pst5>`
shows the zero-filled recipe and plots the RR plane.

Multiplex coherence selection requires the corresponding acquisition
conventions to be represented by ``nmrglue.interop``.

For the MQMAS conventions and broader context, see C. Fernandez and M. Pruski,
*Probing quadrupolar nuclei by solid-state NMR spectroscopy: recent advances*
(2011), `doi:10.1007/128_2011_141 <https://doi.org/10.1007/128_2011_141>`_.
