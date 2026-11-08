Quickstart
==========

.. warning::

   **Experimental plugin.** The public API is still evolving. Function
   signatures, options, and behavior may change between releases, including
   in workflows that have already been validated. Check the :doc:`/changelog`
   before upgrading.

Read a TopSpin FID and process it with explicit, validated 1D parameters:

.. code-block:: python

   import spectrochempy as scp

   fid = scp.nmr.read("path/to/experiment/1/fid")
   spectrum = scp.nmr.Experiment(fid).process(
       apodization="em", lb=2.0, size=32768
   )
   spectrum.plot()

The source dataset is not mutated. Vendor processing metadata is descriptive;
it is not silently replayed by ``Experiment.process()``.
