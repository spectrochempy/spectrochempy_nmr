Installation
============

Install the core and its NMR extra. SpectroChemPy 1.2.1 is the
recommended starting point for the published NMR workflows.

.. code-block:: console

   python -m pip install "spectrochempy[nmr]>=1.2.1"

If you use SpectroChemPy 1.2.0, install ``spectrochempy-nmr`` directly.
The ``nmr`` extra in that version selects the plugin's 0.2 series.
See :doc:`/compatibility` for version requirements. For workflows marked as
available in development, install the plugin from a source checkout as
described in the :doc:`developer guide </devguide/processing_conventions>`.

The plugin stores associated 2D branches as
:doc:`native complex channels </concepts/hypercomplex>`.
