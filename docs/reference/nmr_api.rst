NMR API
-------

The following public objects are owned by ``spectrochempy-nmr``.

.. currentmodule:: spectrochempy_nmr

The recommended reader is exposed after plugin discovery as
``scp.nmr.read(...)``. It dispatches to the concrete reader functions below.
The dispatch function itself is documented in the user guide because it is a
plugin-proxy entry point rather than a module-level function.

``scp.nmr.read`` accepts TopSpin/Bruker, Agilent/Varian, JEOL, TecMag and
SIMPSON inputs and dispatches to the reader functions below. It is the
recommended public entry point even though the callable is created lazily by
the plugin proxy.

.. autoclass:: Experiment
   :members:

.. automodule:: spectrochempy_nmr.readers.read_topspin
   :members: read_topspin

.. automodule:: spectrochempy_nmr.readers.read_agilent
   :members: read_agilent

.. automodule:: spectrochempy_nmr.readers.read_jeol
   :members: read_jeol

.. automodule:: spectrochempy_nmr.readers.read_tecmag
   :members: read_tecmag

.. automodule:: spectrochempy_nmr.readers.read_simpson
   :members: read_simpson

.. automodule:: spectrochempy_nmr.units
   :members: set_nmr_context

Experimental processing
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: spectrochempy_nmr.reference_agilent_1d

.. autofunction:: spectrochempy_nmr.process_agilent_noesy_states_tppi

.. autofunction:: spectrochempy_nmr.extract_noesy_components

.. autofunction:: spectrochempy_nmr.reference_interop_1d
