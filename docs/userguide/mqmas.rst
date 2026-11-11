Processing MQMAS spectra
========================

.. note::

   The staged MQMAS workflow is available in the development version of
   the plugin. See the :doc:`changelog <../changelog>` for release status.

The XFSHEAR workflow separates isotropic and second-order quadrupolar
contributions in MQMAS spectra. It uses a Bruker States acquisition with a
common F2 phase and isotropic F1 calibration.

Transform F2 and inspect the first row
--------------------------------------

Read the SER and supply the F2 FFT size.
For PST-5, keep 2048 acquired complex samples in F2 and zero-fill to
4096 before XF2. This samples the same spectrum on a finer grid before
phase adjustment and quadrature reconstruction. The acquisition interval
and the physical resolution are set by the retained samples.

.. code-block:: python

   import spectrochempy as scp

   raw = scp.nmr.read("nmrdata/bruker/tests/nmr/MQMAS_27AL_PST5/51/ser")
   xf2 = scp.nmr.Experiment(raw).xf2(size=4096, direct_points=2048)
   first_row = xf2[0, 0, :].squeeze()
   phase = {"phc0": -58.0, "phc1": 0.0, "pivot": 0}
   first_row.pk(**phase).real[65.0:20.0].plot()

By default, ``xf2()`` corrects the digital-filter delay before the phase
preview. Adjust the phase on the first-row preview before running XFSHEAR.
The phase angles are in degrees and the pivot is an F2 sample index.
The sign convention matches SpectroChemPy's ``pk()``. To reproduce
TopSpin phase values with a pivot of zero, reverse the signs of PHC0
and PHC1. Keep the original complex XF2 dataset for the next step.

Shear and transform F1
----------------------

Apply the selected phase with ``xf2p()`` to every F1 increment and both
States branches. Pass the phased dataset and the F1 FFT size to ``xfshear()``.
The shear includes the F1 transform.

.. code-block:: python

   phased = scp.nmr.Experiment(xf2).xf2p(**phase)
   spectrum = scp.nmr.Experiment(phased).xfshear(
       size=512, spin=2.5, coherence_order=3, f1_shift=-20.0
   )
   rr = spectrum[0, :, :].squeeze().real
   rr[52.0:25.0, 65.0:20.0].plot(method="contour")

To revise the F2 phase, return to the complex XF2 dataset and run
``xf2p()`` again with the new angles. Use ``pk()`` on the preview row
to select the angles and ``xf2p()`` on the complete dataset to record them
for XFSHEAR. Use ``f1_shift`` to move the signal within the F1 window, in isotropic ppm.
For shifted-echo acquisitions, ``echo_window=(slope, sigma)`` in
``xf2()`` applies a Gaussian along F2 before its FFT.
Its center follows the expected echo at each F1 increment and includes
the digital group delay. Choose the slope for the acquisition and give
``sigma`` in seconds. For baseline correction, fit the selected real plane
along F2 using signal-free regions.

The returned channels pack RR and IR in ``S_R``, and RI and II in
``S_I``. The F2 axis retains the acquisition reference. The F1 axis
uses the isotropic scaling and any requested F1 shift.

The :doc:`PST-5 gallery notebook </examples/gallery/nmr/plot_replay_mqmas_pst5>`
shows the phase preview and compares RR maps with apodization and baseline
correction. See the :ref:`MQMAS qualification notes <mqmas-qualification>`
for the TopSpin comparisons.

For the MQMAS conventions and broader context, see C. Fernandez and
M. Pruski, *Probing quadrupolar nuclei by solid-state NMR spectroscopy:
recent advances* (2011),
`doi:10.1007/128_2011_141 <https://doi.org/10.1007/128_2011_141>`_.
