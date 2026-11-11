Processing MQMAS spectra
========================

The XFSHEAR workflow separates isotropic and second-order quadrupolar
contributions in MQMAS spectra. It uses a Bruker States acquisition with a
common F2 phase and isotropic F1 calibration.

Transform F2 and inspect the first row
--------------------------------------

Read the original SER without DC or digital-filter correction. Supply
the final FFT sizes in F1, F2 order. For PST-5, retain the first 2048
complex samples in F2 and zero-fill to 4096 to meet the TopSpin Hilbert
recommendation.

.. code-block:: python

   raw = scp.nmr.read(
       "nmrdata/bruker/tests/nmr/MQMAS_27AL_PST5/51/ser",
       raw=True,
   )
   xf2 = scp.nmr.Experiment(raw).prepare_xfshear(
       size=(512, 4096), direct_points=2048
   )
   first_row = xf2[0, 0, :].squeeze()
   phase = {"phc0": -58.0, "phc1": 0.0, "pivot": 0}
   first_row.pk(**phase).real[65.0:20.0].plot()

The reader keeps the original samples. XF2 compensates the recorded
group delay on its frequency grid before the phase preview. Use
``correct_digital_filter=False`` to disable that correction explicitly. Adjust the phase on the preview until the region
of interest is phased.
The phase angles are in degrees and the pivot is an F2 sample index.
The sign convention matches SpectroChemPy's ``pk()``. To reproduce
TopSpin phase values with a pivot of zero, reverse the signs of PHC0
and PHC1. Keep the original complex XF2 dataset for the next step.

Shear and transform F1
----------------------

Pass the selected phase to XFSHEAR. It applies that phase to every F1
increment and both States branches before taking the real F2 traces.
The algorithm rotates the States pairs in the hybrid domain, reconstructs
F2 quadrature by Hilbert transform and performs the F1 FFT internally.

.. code-block:: python

   spectrum = scp.nmr.Experiment(xf2).replay_xfshear(
       spin=2.5, coherence_order=3, f1_shift=-20.0, **phase
   )
   rr = spectrum[0, :, :].squeeze().real
   rr[52.0:25.0, 65.0:20.0].plot(method="contour")

To revise the F2 phase, return to the complex XF2 dataset and run
XFSHEAR again. The optional ``f1_shift`` uses the AU convention in
isotropic ppm. It adds a phase ramp before the F1 transform and adjusts
the F1 reference so you can reposition the signal within the window.
For shifted-echo acquisitions, ``echo_window=(slope, sigma)`` in
``prepare_xfshear()`` applies a Gaussian along F2 before its FFT.
Its center follows the expected echo at each F1 increment and includes
the digital group delay. Choose the slope for the acquisition and give
``sigma`` in seconds. For baseline correction, fit the selected real plane
along F2 using signal-free regions.

The returned channels pack RR and IR in ``S_R``, and RI and II in
``S_I``. The F2 axis retains the acquisition reference. The F1 axis
uses the isotropic scaling and any requested F1 shift.

Conventions and qualification
-----------------------------

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

The :doc:`PST-5 gallery notebook </examples/gallery/nmr/plot_replay_mqmas_pst5>`
shows the phase preview and compares RR maps with apodization and baseline
correction. Multiplex coherence selection is a separate acquisition workflow.

For the MQMAS conventions and broader context, see C. Fernandez and
M. Pruski, *Probing quadrupolar nuclei by solid-state NMR spectroscopy:
recent advances* (2011),
`doi:10.1007/128_2011_141 <https://doi.org/10.1007/128_2011_141>`_.
