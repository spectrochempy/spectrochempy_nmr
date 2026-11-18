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
These operations consume normalized States, States-TPPI or echo/anti-echo components, physical acquisition
widths and point counts, frequency references, the observed isotope and the
recorded digital-filter delay. Missing physical references are rejected;
processing does not reread the source parameters or choose a kernel from the
file profile. Read native SCP saves through ``scp.nmr.read()`` to complete
missing physical descriptors in older States saves at the reader boundary.
Choose ``experiment="3QMAS"``, ``"5QMAS"`` or ``"STMAS"`` to compute
coefficients from the normalized spin metadata or observed isotope. An explicit
``spin`` is needed only if both are missing. Without ``experiment``, the
existing MQMAS order selection remains available (3Q by default). STMAS uses
the inner satellite; for spins 3/2, 5/2, 7/2 and 9/2 the ratios are -8/9,
7/24, 28/45 and 55/72, with scale divisors 17/9, 17/24, 17/45 and 17/72.

``experiment="ZQ-TROSY"`` uses ratio -1 and scale 1, without requiring a spin.
As in the `TopSpin AU source <https://www.pascal-man.com/pulseprogram/avance3/xfshear-topspin.shtml>`_,
this branch retains the normalized F1 reference, which can differ from F2.
Its shift parameter uses the F2 carrier; it does not re-reference F1.
Advanced users can override the coefficients with ``ratio`` and ``f1_scale``
together, without supplying spin or coherence order.

Quadrature is independent of this geometry selection: States-TPPI alternation
and echo/anti-echo Rance–Kay conversion are applied once in XF2, then all
experiments share reconstruction, weighting, F1 FFT and the four-component
output. Synthetic equivalence covers all three quadratures. Qualification
on real echo/anti-echo XFSHEAR data remains open; no acquisition is available.
ZQ-TROSY also has synthetic validation, without a real-data qualification.

The convenience ``Experiment.stmas_shear()`` uses automatic STMAS coefficients
and retains its historical shift direction by mapping it to the negative
common shift. It now returns the same four-component representation, rather
than the former complex F1 map. No vendor or pulse-program name selects the kernel.
Read States pairs through ``read()``; an explicit ``indirect_dwell`` selects
the F1 time calibration and ``use_tdeff=True`` selects complete traces from
the processing status. See :doc:`/userguide/stmas`.
Multiplex SPAM uses the same entry points: ``read(path, phase_cycle="paired")``
retains all excitation phases and scan pairs, then
``Experiment(raw).select_coherence(3)`` (or 5 for eleven phases) returns States
E1/E2 FIDs. Continue with XF2, phase and XFSHEAR; automatic shearing inherits
this selection. See :doc:`/userguide/multiplex_spam`.

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
