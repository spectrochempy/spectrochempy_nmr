Installation
============

For ``spectrochempy-nmr`` 0.3.0, install SpectroChemPy 1.2.0 or later
to use the 1D and supported 2D workflows in this guide.

.. code-block:: console

   python -m pip install \
       "spectrochempy>=1.2.0" \
       "spectrochempy-nmr==0.3.0"

Install the plugin explicitly as shown above. The ``nmr`` extra in
SpectroChemPy 1.2.0 selects the plugin's 0.2 series. See
:doc:`/compatibility` for supported core versions and workflow requirements.

Development checkout
--------------------

.. code-block:: console

   python -m pip install -e .

The plugin stores associated 2D branches as
:doc:`native complex channels </concepts/hypercomplex>`.
