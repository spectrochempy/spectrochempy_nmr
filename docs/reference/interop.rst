Interoperability and referencing
================================

Convert normalized nmrglue-ng results to SpectroChemPy datasets and
work with NMR frequency references. For exchange files, start with
the :ref:`public reader <reading-exchange-formats-through-interop>`.

Raw 2D component contract
-------------------------

For qualified QF, States, States-TPPI, TPPI and echo/anti-echo inputs,
``read()`` and ``from_interop()`` arrange the declared components and retain
the physical axes and acquisition widths. Explicit component labels determine
the ordering: permutations of the same complete label set produce the same
native array. A selected States input convention is converted at this boundary,
with the operation recorded in the dataset metadata.

``Experiment(dataset).process()`` selects the Fourier reconstruction from
quadrature and component meaning. It does not select an algorithm from the
vendor, file profile or experiment name. A saved SCP dataset retains this
information and can be processed without reopening the original file.
Echo/anti-echo inputs use the normalized Rance–Kay arrangement by default;
``echo_antiecho="rance-kay"`` is optional, and the historical
``"bruker-rance-kay"`` spelling remains an alias. Digital-filter correction
uses the imported delay descriptor or an explicit caller override, rather
than rereading vendor parameters during this common processing path.

An undetermined physical quadrature remains unsupported for a complete
indirect transform. Importing stored components does not establish that
quadrature. This contract does not qualify additional file formats or modes.
Recorded TopSpin replay and specialized experimental helpers remain separate
entry points.

.. currentmodule:: spectrochempy_nmr

.. autosummary::
   :nosignatures:
   :toctree: generated

   from_interop
   set_nmr_context

Export processed spectra
-------------------------

See :doc:`/userguide/exchanging_spectra` for the supported components,
axis units and round-trip examples.

.. autosummary::
   :nosignatures:
   :toctree: generated

   write_jcamp
   write_csdf

Experimental entry points
-------------------------

.. autosummary::
   :nosignatures:
   :toctree: generated

   read_agilent_interop
   reference_agilent_1d
   reference_interop_1d
