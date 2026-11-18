Experiment and processing
=========================

Use ``scp.nmr.Experiment(dataset)`` to inspect the acquisition and run a
processing workflow. See :doc:`/userguide/processing_1d` and
:doc:`/userguide/processing_2d` for examples.
States and States-TPPI use the same processing entry point regardless of
experiment name. When the acquisition does not establish the quadrature,
select it explicitly at reading, for example for a supported Agilent pair::

    fid = scp.nmr.read("path/to/acquisition", quadrature="states-tppi")
    spectrum = scp.nmr.Experiment(fid).process()
    rr = scp.nmr.Experiment(spectrum).RR

The four real output planes use physical F1/F2 labels. ``Experiment.RR``,
``RI``, ``IR`` and ``II`` follow the TopSpin component convention; use these
accessors for component extraction.

For the Bruker States MQMAS workflow, see :doc:`/userguide/mqmas`.
Use ``Experiment.xf2()`` to transform F2, ``Experiment.xf2p()`` to apply
the common phase, and ``Experiment.xfshear()`` to produce the final spectrum.
These operations consume normalized States components, physical acquisition
widths and point counts, frequency references, the observed isotope and the
recorded digital-filter delay. Missing physical references are rejected;
processing does not reread the source parameters or choose a kernel from the
file profile. Read native SCP saves through ``scp.nmr.read()`` to complete
missing physical descriptors in older States saves at the reader boundary.
For spin-3/2 DQF-STMAS, use ``Experiment.stmas_shear()`` after the common
phase. The operation checks the normalized observed isotope for spin 3/2;
no reader-specific STMAS marker or pulse-program name selects the calculation.
Read States pairs through ``read()``; an explicit ``indirect_dwell`` selects
the F1 time calibration and ``use_tdeff=True`` selects complete traces from
the processing status. See :doc:`/userguide/stmas`.
For polynomial baseline correction along F1 or F2 and numbered trace
extraction, see :ref:`topspin-slice-and-baseline`.

.. currentmodule:: spectrochempy_nmr

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/experiment.rst

   Experiment

Automatic phase estimation
--------------------------

.. autosummary::
   :nosignatures:
   :toctree: generated

   autophase
