Installation
============

This documentation targets ``spectrochempy-nmr`` 0.2.0 and requires
SpectroChemPy 1.1 or later, below core 2.0.

.. code-block:: console

   python -m pip install \
       "spectrochempy>=1.1.0,<2" \
       "spectrochempy-nmr==0.2.0"

Development checkout (Unreleased)
---------------------------------

.. code-block:: console

   python -m pip install -e .

The development plugin stores associated 2D branches as ordinary complex
arrays. No Hypercomplex or numpy-quaternion package is required.
See :doc:`/concepts/hypercomplex` for the new storage contract and
:doc:`/compatibility` for scope.
