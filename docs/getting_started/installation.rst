Installation
============

This documentation targets ``spectrochempy-nmr`` 0.2.0 and requires
SpectroChemPy 1.1 or later, below core 2.0.

.. code-block:: console

   python -m pip install \
       "spectrochempy>=1.1.0,<2" \
       "spectrochempy-nmr==0.2.0"

Install hypercomplex representation separately when needed:

.. code-block:: console

   python -m pip install \
       "spectrochempy>=1.1.0,<2" \
       "spectrochempy-nmr==0.2.0" \
       "spectrochempy-hypercomplex==0.2.0"

See :doc:`/compatibility` for the exact local build
environment. Hypercomplex support does not depend on the NMR plugin.
