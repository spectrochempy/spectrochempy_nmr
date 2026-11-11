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
   padded = windowed.zf_size(size=max(32768, windowed.shape[-1]))
   spectrum = padded.fft()

Both approaches preserve the original FID by default. A supported complex
JEOL 1D FID read with ``scp.nmr.read_jeol()`` can use either approach.
For example, on the epicatechin JDF reference, ``Experiment.process()``
produces a spectrum referenced in ppm, and the stepwise core workflow above
produce a frequency axis in Hz.

For a 1D acquisition outside the scope of ``Experiment.process()``, you
can still apply individual operations when the data meet their input
requirements. For example, use a suitable time-domain axis for apodization
and FFT.

An exponential window trades some resolution for sensitivity. Zero filling
sets the spacing of the frequency bins. Once the spectrum is in the
frequency domain, use ``spectrum.pk(phc0=..., phc1=...)`` to apply explicit
phase angles in degrees. Plot the result with ``spectrum.plot()``, then use
:doc:`analysis` for baseline correction, peak picking, integration and
fitting.

A direct ``.fft()`` transforms the samples supplied to it. For a Bruker FID
with a supported preparation descriptor, use
``experiment.prepare_fid(digital_filter="time")`` before ``.fft()``, or
``experiment.process(digital_filter="time")`` to correct the filter as part
of processing. To subtract the DC offset as well, pass
``remove_dc_offset=True`` to ``prepare_fid()``. To reproduce a script that
previously relied on correction during import, pass
``remove_digital_filter=True`` to the reader.

A pseudo-2D series contains ordinary 1D spectra that can be processed one
spectrum at a time. For acquisitions with two Fourier dimensions, see
:doc:`processing_2d`.

See :ref:`nmr-core-operations` for the core functions used in NMR.

Estimate phase and correct the baseline
---------------------------------------

The standalone ``autophase()`` workflow accepts a complex spectrum after the
FFT and requires a version of nmrglue-ng that provides ``estimate_phase()``.

.. code-block:: python

   from spectrochempy_nmr import autophase

   diagnostic = autophase(spectrum, order=0, apply=False)
   result = autophase(spectrum, order=0, baseline_options={"model": "asls", "lamb": 1e9})

Inspect ``result.estimate`` for the correction angles and fit diagnostics.
``result.phased`` is the complex spectrum. If you request baseline correction,
``result.baseline`` and ``result.corrected`` are real datasets. Use
``result.phased`` to make further phase adjustments. Its phase metadata records
the correction relative to the input of this call, and the history retains
the previous phase metadata. For baseline fitting, values behind the dataset
mask are filled from neighboring valid samples. The output retains the
original mask. AsLS is intended here
for spectra with positive peaks. Tune its smoothness for the spectral grid
and check that integration areas are preserved. For signed spectra, use a polynomial fit on
known signal-free regions with ``baseline_options`` specifying ``ranges``.

ACME assumes positive absorption peaks. Pass a boolean ``exclude`` array to
omit artifacts from phase scoring while retaining their data in the result.
For isolated peaks of known signs, choose ``method="dispa"`` and provide
``references=[(start, stop, sign), ...]`` with half-open index windows. Use
well-separated references for order one. The optional
``baseline_mask`` selects signal-free indices to fit a provisional complex
linear background for phase scoring. The final real baseline is a separate
fit. A successful solver status still calls for inspecting the result.

For a spectrum produced with ``fftshift(fft(fid))`` on an increasing
frequency grid, use the measured receiver timing to fix the slope and fit
only the constant phase. A rectangular hard 90° pulse contributes
approximately twice its duration divided by pi to the effective delay.
Use the time from the pulse end to the first complex sample and account for
any digital filter correction separately.

.. code-block:: python

   from nmrglue.process.proc_autophase import phase_slope_from_delay

   slope = phase_slope_from_delay(10_000, 8e-6, hard_90_pulse_s=10e-6)
   result = autophase(spectrum, fixed_p1=slope)

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

Fractional PKNL correction for a Bruker QSIM FID
------------------------------------------------

Read the original FID with digital-filter and DC correction disabled.
The plugin obtains the group delay from the acquisition and applies its
fractional phase ramp on the final FFT grid. It rejects a FID whose
digital filter was already corrected at import.

.. code-block:: python

   fid = scp.nmr.read_topspin(
       "experiment/80/fid", remove_digital_filter=False, remove_dc_offset=False
   )
   spectrum = scp.nmr.Experiment(fid).process(
       apodization="em", lb=50.0, size=8192, digital_filter="pknl"
   )

The result uses the processing ppm reference. To apply measured phase
angles at the first frequency bin, pass ``phase="manual"`` with ``phc0``
and ``phc1`` in degrees. The
:doc:`cross-polarization example </examples/gallery/analysis/plot_processing_cp_nmr>`
uses one pair of angles from a representative FID for the entire series.
