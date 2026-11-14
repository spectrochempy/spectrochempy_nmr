Nuclear isotopes
~~~~~~~~~~~~~~~~

``scp.nmr.Isotope`` reads the isotope table shipped with SpectroChemPy.
Use the isotope label to look up the exact nuclear spin, electric
quadrupole moment, signed gyromagnetic ratio and natural abundance.

.. code-block:: python

   import spectrochempy as scp

   aluminium = scp.nmr.Isotope("27Al")
   spin = aluminium.spin                  # Fraction(5, 2)
   larmor = aluminium.larmor_frequency(9.4) # MHz at 9.4 T

The tabulated quadrupole moment is a nuclear property. A sample's
quadrupolar coupling also depends on its electric field gradient.

.. autoclass:: spectrochempy_nmr.Isotope
   :members:
