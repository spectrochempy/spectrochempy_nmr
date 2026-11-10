Processing 2D NMR data
======================

Use the plugin's :doc:`reader <reading_data>` to load the 2D dataset.
The acquisition and quadrature determine which operations are available.
See :doc:`/concepts/quadrature` for the supported conventions and
:ref:`nmr-processing-validation` for the distinction between runtime checks
and qualification against reference data.

When a TopSpin read includes a ``procs`` file, the dataset can expose its
processing parameters in ``dataset.meta.nmr_processing["vendor_profile"]``.
The profile records the vendor, the source file, and the parameter values.
Replay requires a supported combination of acquisition and processing
settings, as described in :ref:`topspin-xfb-replay`.

For a supported uniform Bruker SER acquisition, use the recorded TopSpin
processing parameters explicitly:

.. code-block:: python

   import spectrochempy as scp

   fid = scp.read_topspin(
       "path/to/experiment/ser",
       remove_digital_filter=False,
       remove_dc_offset=False,
   )
   spectrum = scp.nmr.Experiment(fid).process(replay="topspin")
   trace = spectrum.meta.nmr_processing["scp_processing"]["applied"]

XFB replay supports indirect QF magnitude, States-TPPI, and echo/anti-echo
with direct DQD. See :ref:`topspin-xfb-replay` for the supported settings
and reference comparisons. The ordinary ``Experiment.process()`` path
applies SpectroChemPy's processing sequence to supported acquisitions.

For the underlying dataset, plotting, and general processing API, see
the `NMR plugin overview in the SpectroChemPy core documentation
<https://www.spectrochempy.fr/latest/userguide/plugins/nmr.html>`_.
