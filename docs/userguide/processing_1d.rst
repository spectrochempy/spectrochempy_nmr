.. _validated-1d-processing:

Processing 1D NMR data
======================

Wrap a dataset in :class:`~spectrochempy_nmr.Experiment` and select explicit
processing parameters:

.. code-block:: python

   experiment = scp.nmr.Experiment(fid)
   result = experiment.process(apodization="em", lb=5.0, phase=None)

The supported apodization modes are ``em``, ``gm`` and ``sp``. The result
records the SpectroChemPy-owned call in
``result.meta.nmr_processing["scp_processing"]``. ``requested`` contains
explicit user arguments and ``applied`` records operations actually executed.

To inspect intermediate results, apply the core operations to the
``NDDataset`` exposed by ``experiment.dataset``. You can call them as
dataset methods or as functions such as ``scp.em(experiment.dataset, lb=5.0)``.

.. code-block:: python

   windowed = experiment.dataset.em(lb=5.0)
   spectrum = windowed.fft()

The original FID is preserved by default in both workflows.
Use ``Experiment.process()`` when you want the plugin to validate the
processing sequence for the acquisition and record the operations it applies.

This workflow also applies to supported complex JEOL 1D FIDs. Read the
``.jdf`` file with ``scp.nmr.read_jeol()``, then use
``Experiment.process()`` or apply the core operations individually.
For other 1D acquisitions, a workflow outside the scope of
``Experiment.process()`` can still be processed step by step when the
dataset meets each operation's requirements. For example, apodization
and FFT require a suitable time-domain axis.

A pseudo-2D series contains ordinary 1D spectra that can be processed one
spectrum at a time. For acquisitions with two Fourier dimensions, see
:doc:`processing_2d`.

See :ref:`nmr-core-operations` for the core functions used in NMR.

Replay TopSpin EFP on a Bruker DQD FID
--------------------------------------

Use the raw FID and the processing status file (``procs``) from a completed
TopSpin ``efp`` run in the same experiment:

.. code-block:: python

   fid = scp.read_topspin("experiment/1/fid", remove_digital_filter=False)
   spectrum = scp.nmr.Experiment(fid).process(replay="topspin")

Replay uses the recorded exponential window, FFT size and phase. It applies
digital-filter compensation on the final FFT grid, including fractional group
delay, and uses the processing ppm axis. Intensities retain the FID reader's
scan and receiver gain normalization, so comparison to TopSpin may require an
overall amplitude factor.

The supported profile is a full-spectrum DQD 1D ``efp`` run with
``PKNL`` enabled, ``FCOR=1``, and baseline correction, linear prediction and
axis reversal disabled.
