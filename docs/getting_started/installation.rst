Installation
============

This documentation targets ``spectrochempy-nmr`` 0.3.0 and requires
SpectroChemPy 1.1 or later, below core 2.0.

.. code-block:: console

   python -m pip install \
       "spectrochempy>=1.1.0,<2" \
       "spectrochempy-nmr==0.3.0"

For qualified 2D workflows that use the core FFT prevalidation hook, use
SpectroChemPy 1.2.0 or later. Install the plugin explicitly with core 1.2.0
because that release's ``nmr`` extra selects the 0.2 series.

Development checkout
--------------------

.. code-block:: console

   python -m pip install -e .

The plugin stores associated 2D branches as ordinary complex
arrays. No Hypercomplex or numpy-quaternion package is required.
See :doc:`/concepts/hypercomplex` for the new storage contract and
:doc:`/compatibility` for scope.
