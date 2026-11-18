Readers
-------

Use ``scp.nmr.read()`` to select a reader by file type or pass a protocol
explicitly. Use the format-specific functions when you already know the
format. See :doc:`/userguide/reading_data` for supported inputs and examples.

.. currentmodule:: spectrochempy_nmr

.. autosummary::
   :nosignatures:
   :toctree: generated

   read
   read_topspin
   read_multiplex_spam
   read_stmas
   read_agilent
   read_jeol
   read_simpson
   read_csdf

JEOL 2D acquisition branches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   The low-level import described here is an incomplete workflow.
   Complete processing of JEOL HSQC, HMQC and HMBC acquisitions is not
   supported by the plugin. For a supported uniform proton COSY workflow,
   use :doc:`/userguide/jeol_cosy_magnitude`.

For a JDF layout accepted by the low-level importer, select interop to keep the stored section
pairs as separate complex components:

.. code-block:: python

   dataset = scp.nmr.read(
       "hsqc.jdf",
       protocol="interop",
       normalize_options={"conjugation": "none"},
   )

Four-section acquisitions use dimensions ``c, y, x`` for the two
components, F1, and F2. Time coordinates are in seconds. For listed F1
sampling, ``dataset.y`` contains the acquired times and
``dataset.meta.acquisition["nominal_shape"]`` records the full sampling grid.
F1 quadrature remains undetermined and F1 Fourier transformation is guarded.
The dedicated ``read_jeol()`` reader retains the historical interleaved
presentation and signs. Uniform four-section files require a provider
with uniform complex/complex JEOL normalization support.
