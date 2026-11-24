Refresh an incomplete example
=============================

Use this page if a downloaded STMAS or Multiplex example reports missing
acquisition or processing parameter files. An older cached copy may contain
only the recorded signal.

Find the downloaded example
---------------------------

In your notebook, display the example-data location:

.. code-block:: python

   import spectrochempy as scp

   print(scp.preferences.datadir)

Open that location in your file manager. The example folders are:

* STMAS: ``nmrdata/bruker/tests/nmr/RbNO3/15``.
* Multiplex SPAM: ``nmrdata/bruker/tests/nmr/RbNO3/206``.

Keep the old copy and download a complete one
---------------------------------------------

Rename the downloaded example folder to ``15-backup`` or ``206-backup`` so its
contents remain available. Rerun the recipe with an internet connection.
The read downloads a new complete folder under the original name, then
continues processing.

These steps apply to the public examples. For your own measurements, keep
the full experiment export from the spectrometer, including its parameter
files; the public example download cannot supply them.
