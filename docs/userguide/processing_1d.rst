.. _validated-1d-processing:

Processing 1D NMR data
======================

.. _nmr-processing-validation:

Processing, validation, and qualification
-----------------------------------------

* **Processing** means applying operations to the data. For example, apply
  exponential apodization and then an FFT to a time-domain ``NDDataset``.
  When working step by step, you choose the operations and their order.
* **Validation** is the set of checks made when you call
  ``Experiment.process()``. The plugin checks whether it can run the
  requested workflow on that dataset, using properties such as its domain,
  dimensions and acquisition encoding. Individual core operations also
  have their own input checks.
* **Qualification** is evidence from testing a specific acquisition and
  processing configuration. For example, the TopSpin EFP replay below was
  compared with a TopSpin reference for the documented Bruker DQD profile.
  That comparison applies to the settings covered by the reference tests.
  A successful call to ``Experiment.process()`` tells you the input passed
  the runtime checks.

Processing a FID
----------------

Wrap a dataset in :class:`~spectrochempy_nmr.Experiment` and select explicit
processing parameters:

.. code-block:: python

   experiment = scp.nmr.Experiment(fid)
   result = experiment.process(apodization="em", lb=5.0, phase=None)

The supported apodization modes are ``em``, ``gm`` and ``sp``. The result
records the workflow in
``result.meta.nmr_processing["scp_processing"]``. ``requested`` contains
explicit user arguments and ``applied`` records operations actually executed.

To inspect intermediate results, apply the core operations to the
``NDDataset`` exposed by ``experiment.dataset``. You can call them as
dataset methods or as functions such as ``scp.em(experiment.dataset, lb=5.0)``.

.. code-block:: python

   windowed = experiment.dataset.em(lb=5.0)
   spectrum = windowed.fft()

Both approaches preserve the original FID by default. A supported complex
JEOL 1D FID read with ``scp.nmr.read_jeol()`` can use either approach.
For example, on the epicatechin JDF reference, ``Experiment.process()``
produces a spectrum referenced in ppm, and the two core calls above
produce a frequency axis in Hz.

For a 1D acquisition outside the scope of ``Experiment.process()``, you
can still apply individual operations when the data meet their input
requirements. For example, use a suitable time-domain axis for apodization
and FFT.

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
