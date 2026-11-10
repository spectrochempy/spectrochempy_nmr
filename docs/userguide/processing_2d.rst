Processing 2D NMR data
======================

Use the plugin's :doc:`reader <reading_data>` to load a qualified 2D dataset.
The acquisition and quadrature determine which operations are available.
See :doc:`/concepts/quadrature` for the supported conventions.

For a qualified uniform Bruker SER acquisition, use the recorded TopSpin
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
processes qualified data without requesting vendor replay.

For the underlying dataset, plotting, and general processing API, see
the `SpectroChemPy core documentation <https://www.spectrochempy.fr/>`_.
