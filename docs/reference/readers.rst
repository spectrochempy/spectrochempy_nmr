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

For Multiplex SPAM, prefer ``read(path, phase_cycle="paired")`` followed by
``Experiment(raw).select_coherence()``. Reading retains acquired phases;
selection and the common XF2 workflow perform reconstruction.
``read_multiplex_spam()`` is a compatibility shortcut. See
:doc:`/userguide/multiplex_spam`.

JEOL 2D acquisition branches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a supported JDF acquisition, select interop to keep the stored section
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
