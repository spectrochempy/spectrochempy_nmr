Interoperability and referencing
================================

Convert normalized nmrglue-ng results to SpectroChemPy datasets and
work with NMR frequency references. For exchange files, start with
the :ref:`public reader <reading-exchange-formats-through-interop>`.

.. currentmodule:: spectrochempy_nmr

.. autosummary::
   :nosignatures:
   :toctree: generated

   from_interop
   set_nmr_context

Experimental entry points
-------------------------

.. autosummary::
   :nosignatures:
   :toctree: generated

   read_agilent_interop
   reference_agilent_1d
   reference_interop_1d
