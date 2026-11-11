Normalized Bruker 1D qualification
==================================

This workflow is **experimental**. It demonstrates that a second
real producer feeds the same normalized ``from_interop`` and ``Experiment``
processing path without modifying the core or adding Bruker processing logic
to the plugin::

   import nmrglue as ng
   from spectrochempy_nmr import from_interop
   from spectrochempy_nmr.experiment import Experiment

   raw = ng.interop.read_bruker_fid("/path/to/acquisition")
   normalized = ng.interop.normalize_bruker_fid(raw, remove_digital_filter=True)
   fid = from_interop(normalized)
   spectrum_hz = Experiment(fid).process(size=32768)
   spectrum_ppm = Experiment(fid).process(size=32768, reference="acquisition")

The qualification requirements pin the nmrglue-ng version used for these tests.
The public TopSpin reader uses the normalized FID contract introduced in
nmrglue-ng PR #197 when the installed provider supports it. For qualified
complex 1D acquisitions, ``read_topspin()`` returns the dataset produced by
``from_interop()``. The provider preserves the decoded length, applies the
requested DC and filter corrections in order, and divides by NS times RG.

Samples use the canonical complex convention. Compared with the historical
TopSpin FID, this reverses the sign of the imaginary channel. Acquisition
quadrature remains available as ``meta.acquisition_encoding``. The default
FFT reproduces the historical spectrum and display axis using the provider's
phase and display descriptors, when available. An explicit
``Experiment(fid).process(reference="acquisition")`` uses the physical FFT
grid and acquisition reference without the historical initial phase.

Providers without the new contract and acquisitions outside its coverage
continue through the previous reader path.

The public reader also delegates complete Bruker 2D SER decoding to
nmrglue-ng when the decoded layout matches the historical shape. The
plugin still applies the public reader's trace pairing, corrections, gain
and axes. Incomplete traces and nonuniform sampling retain the historical
reader path.

The direct normalization example trims zero padding to TD/2, applies the
existing truncated group-delay correction, and conjugates into the canonical
DFT orientation. It preserves the original samples and raw metadata in the
input result and records the transformations in ``applied_operations``.
The neutral descriptor supplies BF1/O1 acquisition referencing. Its opaque
``acquisition`` candidate selects the physical reference. The direct example does not replay
the processed-data reference or the initial phase. Experiment uses normalized
properties for these choices, independent of origin or profile identity.

Public QF SER reader
--------------------

For a complete, uniformly sampled QF SER with calibrated processing axes,
``read_topspin()`` can return the dataset produced by ``from_interop()``.
The installed provider must support per-trace corrections and the QF FFT
presentation descriptor. The core must validate FFT order before mutating
the dataset (``FFT_PREVALIDATION_API >= 1``). The returned samples retain the stored direct
complex convention. Their imaginary component therefore has the opposite
sign from the historical reader output. The provider applies selected
corrections and scan/gain scaling once.

The historical ``dataset.fft(dim="x").fft(dim="y")`` presentation is retained
using the declared transforms, initial phases and decreasing ppm axes.
Transform F2 before F1. The reader rejects an F1 transform while F2 is
still in the time domain.
``Experiment(dataset).process()`` uses the separately qualified physical QF
FFT convention and carrier-relative Hz axes. SERs outside this qualified
path continue through the previous reader.

Public States SER reader
------------------------

For complete uniform Bruker States SERs with direct DQD and calibrated
processing axes, ``read_topspin()`` selects States from ``acqu2s/FnMODE=4``
and returns the normalized E1/E2 dataset. A provider with the States
normalizer and a core with FFT prevalidation are required. The public
reader preserves aligned trace length and applies the selected corrections
before pairing. Other storage profiles continue through the earlier reader.

Transform F2 before F1. States uses the same cross-pairs as States-TPPI
without sign alternation. Public FFTs use the ppm presentation. Apply
any additional phase with ``pk()``. ``Experiment(dataset).process()`` uses the NMRPipe
FFT convention and carrier-relative Hz axes.

To replay the opt-in NMRPipe comparison, set
``NMR_BRUKER_PUBLIC_STATES_DATA`` to the pinned reference directory.
Put ``nmrPipe`` and ``bruk2pipe`` on ``PATH``, or point
``NMR_PIPE_EXECUTABLE`` and ``NMR_BRUK2PIPE_EXECUTABLE`` at them.
Run ``pytest tests/qualification/test_bruker_states_public.py``.

Public States-TPPI SER reader
-----------------------------

For complete uniform States-TPPI SERs with both processing axes calibrated,
``read_topspin()`` can return the normalized E1/E2 dataset from
``from_interop()``. This requires a provider with the States-TPPI storage
normalizer and a core with FFT prevalidation. The reader selects the
aligned trace length for compatibility. The normalizer's default instead
uses only acquired points. DC, filter and scan/gain options apply once
before pairing. The samples retain their stored complex orientation.

Transform F2 before F1. The first public FFT retains the encoded alternation
and the second applies it once when transforming the cross-pairs. Each axis
uses the provider's phase and ppm display parameters. The generic
``Experiment.process()`` route retains its NMRPipe-qualified convention.
Unsupported acquisitions continue through the earlier reader.

Public echo/anti-echo SER
-------------------------

For a complete, uniform Bruker echo/anti-echo acquisition with calibrated
FFT display parameters, the public reader returns the dataset produced by
``from_interop()``. As with States-TPPI, the public reader preserves aligned
trace length and applies the selected corrections once before pairing.

The public FFT transforms F2 before F1 and reproduces the historical
branchwise transform, including the initial phase and ppm display convention.
The stored branches can also be processed with
``Experiment.process(echo_antiecho="bruker-rance-kay")`` for physical
Rance–Kay reconstruction. The two routes have distinct reconstruction
metadata. Profiles without the required calibration retain the historical
reader.

Public Bruker 2D processed spectra
----------------------------------

For complete four-plane int32 Bruker pdata with calibrated axes, the public
reader returns the interop dataset with two complex channels. The provider
pairs the stored planes, applies the receiver gain once and selects either
the acquisition-centered axis or the processing axis requested with
``proc_axis=True``. The latter uses SF for ppm calibration while SFO1 remains
available as acquisition metadata. The four stored planes remain available
through the provider's default normalization. The historical reader
continues to cover profiles outside this contract and custom list axes.

Decoding outside the normalized profiles
----------------------------------------

TopSpin uses nmrglue-ng for both qualified spectra and compatibility reads.
The latter retain the public treatment of incomplete acquisitions, including
the trace length used when recovering flat SER data. Nonuniform sampling is
retained in acquisition metadata. Loading the samples does not reconstruct
the NUS schedule or qualify that acquisition for a normalized FFT.

Prepare a FID explicitly
------------------------

With a provider that exposes the deferred preparation contract, leave the
normalizer's correction options off and apply selected operations to a copy
before processing::

   raw = ng.interop.read_bruker_fid("/path/to/acquisition")
   fid = from_interop(ng.interop.normalize_bruker_fid(raw, preserve_storage=True))
   prepared = Experiment(fid).prepare_fid(
       remove_dc_offset=True, digital_filter="time", normalize_scan_gain=True
   )
   spectrum = Experiment(prepared).process(size=32768, reference="acquisition")

The method applies complex-mean subtraction, the declared time-domain filter
and scan/gain normalization in that order. Mean subtraction can suppress
signal at the carrier. Existing corrections are retained. Requests that
would put a missing correction out of order raise an error. A filter is
skipped when the provider declares that it is unnecessary. Missing evidence
for a requested correction raises an error. Import and the public TopSpin
reader retain their existing behavior.

Qualified local acquisitions
----------------------------

The two originals remain outside both repositories:

* Historical nmrglue Bruker 1D: DQD, big-endian int32, 2048 complex points,
  10000 Hz width, BF1=600.13 MHz, O1=2820.61100006104 Hz. DSPFVS 12/DECIM 16
  supplies delay 71.625, truncated to 71 with 73 points removed.
* SCP TopSpin 1D: QSIM, little-endian int32, 12544 stored complex points,
  TD/2=12487 acquired points (57 zero padding points), 250000 Hz width,
  BF1=500.07 MHz, O1=3117.4 Hz. GRPDLY 76 removes 78 points after padding trim.

Run all local cases with the installed candidate/provider wheels::

   export NMR_BRUKER_LEGACY_DATA=/path/to/bruker_1d
   export NMR_BRUKER_TOPSPIN_DATA=/path/to/topspin_1d/1
   python -I -m pytest -ra tests/qualification/test_bruker_normalized.py

Configured missing files or mismatched SHA-256 checksums fail. Unconfigured
real cases are skipped and do not count as qualification. The module separately
includes two self-contained binary cases generated in temporary directories;
CI executes those with the installed provider. They are synthetic, not extra
real acquisitions. No real data or derived spectra are redistributed.

Tests independently decode the original int32 binary, verify acquired/storage
sizes, time axes, complex spectra, ppm spacing, metadata independence, source
immutability and SCP persistence. Corrected cases use a direct integer shift
and documented head/tail fold oracle. No RG/NS amplitude calibration is applied.

External numerical oracle
-------------------------

The optional tool compares unfiltered results with NMRPipe 13. All OBS/CAR,
size, byte order and spectral-width arguments are explicit oracle inputs::

   python tools/qualify_bruker_normalized.py /path/to/bruker_1d \
     --nmrpipe-bin /path/to/nmrpipe/bin --points 2048 --storage-points 2048 \
     --spectral-width 10000 --observation-mhz 600.13 \
     --carrier-ppm 4.70000000010171 --size 4096 --mode DQD --big-endian

   python tools/qualify_bruker_normalized.py /path/to/topspin_1d/1 \
     --nmrpipe-bin /path/to/nmrpipe/bin --points 12487 --storage-points 12544 \
     --spectral-width 250000 --observation-mhz 500.07 \
     --carrier-ppm 6.233927250184974 --size 32768 --mode Complex

The tool uses ``bruk2pipe -AMX`` and ``FT -nodmx``; digital-filter removal is
disabled in both paths. The canonical complex FFT equals the conjugate of the
NMRPipe spectrum. The measured relative errors are below 2e-7, and maximum ppm
errors below 1e-5 ppm (tolerance 2e-5 for float32 NMRPipe headers). These results
do not qualify NMRPipe's separate digital-filter algorithm, absolute phase or
physical amplitude calibration. Both original files are hash-checked before and
after; all generated files stay in temporary directories.

Local TopSpin example acquisitions
----------------------------------

Set ``NMR_TOPSPIN_EXAMDATA`` to the installed TopSpin ``examdata``
directory and run ``pytest tests/qualification/test_topspin_examdata.py``.
The experiment-folder regression in ``test_bruker_ser_migration.py`` uses
a synthetic acquisition and runs
without the corpus. With ``bruk2pipe`` and ``nmrPipe`` on ``PATH``, the
uniform 2D cases also compare the carrier-relative FFT planes to NMRPipe.
Temporary transport copies are used for conversion. The installed data
are read in place.

The NUS cases characterize storage and the current refusal to run a
uniform FFT. The Ubiquitin case checks the provider's native 3D storage
layout; public TopSpin processing is currently qualified for 1D and
supported 2D profiles.

.. _topspin-xfb-replay:

TopSpin XFB replay
------------------

Read the uniform Bruker SER with ``remove_digital_filter=False`` and
``remove_dc_offset=False``, then call
``Experiment(dataset).process(replay="topspin")``. Replay uses the recorded
processing status of both axes. The supported profiles cover direct DQD
with indirect QF magnitude, States-TPPI, or Bruker echo/anti-echo. They
support no window, exponential (``WDW=1``), sine (``WDW=3``), and
squared sine (``WDW=4``) on either axis. Replay uses the recorded
``LB`` in hertz for the exponential, including signed values, and
``SSB`` for both sine windows. In F2, the group delay rounded up to the
next sample sets the window origin with either setting of ``PKNL``.
When enabled, ``PKNL`` uses the full delay for its phase ramp. Replay also supports
F1 first-point weighting with ``FCOR=1`` on F2, zero filling, phase,
and F2 ``PKNL``.
The F1 status can also request forward linear prediction (``ME_mod=2``,
TopSpin ``LPfr``). Replay predicts complex traces with the Burg recurrence
after F2 processing and before the F1 window and first-point weighting.
An even ``NCOEF``, at most the acquired F1 increment count, specifies
twice the complex predictor order. ``LPBIN``
and ``TDeff`` must agree and describe the extended length in F1 scalar words
(one per QF increment, two per paired increment). The extended length must
fit within F1 ``SI``. Replay rejects a requested order if the residual energy
of a nonzero trace falls below floating-point precision during the fit.
The result uses the processing ppm
axes and retains the FID scan/gain normalization.

The QF result is a magnitude matrix. The two complex channels of a
phase-sensitive result use the same component layout as the public
TopSpin processed reader. The real part of the first channel is
``2rr``. Replay rejects F2 prediction, F1 prediction modes other than
``LPfr``, time-domain offsets (``TDoff``), reversed axes, and extracted
processing grids.

The ``BC_mod`` status describes correction of the time-domain traces.
On F1, ``single`` (1) and ``quad`` (2) subtract the mean of the last
quarter and require a whole number of quarters. ``spol`` (3) and
``qpol`` (4) subtract a degree-five polynomial. The single-channel modes
use TopSpin's interleaved scalar order. The polynomial replay requires
at least six increments. ``qpol`` supports a finite ``COROFFS`` in hertz,
while ``spol`` requires zero. On F2, the DQD processing status records
``single`` and ``quad`` requests as off, and ``spol`` as ``qpol``.
F2 ``qpol`` fits and corrects the points starting at the rounded-up group delay.
Corrections precede the window and linear prediction on their axis.
``sfil`` (5) and ``qfil`` (6) use a Gaussian convolution with linear
extrapolation at the trace ends. Both require positive finite ``BCFW``
in ppm and at least eight increments. F1 ``sfil`` operates on the
interleaved scalar words with zero ``COROFFS``. F1 and F2 ``qfil``
support a finite ``COROFFS`` in hertz. DQD records an F2 ``sfil``
request as ``qfil``, which operates from the same rounded-up origin.

To run the external comparisons, set ``NMR_TOPSPIN_XFB_REFERENCES`` to
a directory containing experiment folders ``20``, ``30`` and ``40``
copied from Menthyl-Anthranilate. Set ``ME_mod=0`` in F1 on each copy
and run ``xfb`` in TopSpin to populate ``pdata/1``. For the prediction
comparisons, set ``NMR_TOPSPIN_XFB_LP_REFERENCES`` to a separate set of the
same experiment folders processed with the original F1 ``LPfr`` settings.
Run
``pytest tests/qualification/test_topspin_examdata.py -k xfb``.

With the same reference directories and the core and plugin installed, run
``python -I -m pytest tests/qualification/test_topspin_xfb_roundtrip.py -k paired`` to
qualify saved replay for experiments ``30`` and ``40``. The test compares
all four component planes and the ppm axes against TopSpin, then checks
exact parity of the spectrum, axes and trace in a separate Python process
with the raw metadata removed and access to the provider and the corpus blocked.

For the window comparisons, set ``NMR_TOPSPIN_XFB_WINDOW_REFERENCES`` to
a directory with six case subdirectories, each containing the same three
experiment folders processed by ``xfb``. Use ``LB=37`` Hz in F1 and
``LB=47`` Hz in F2 wherever the exponential is selected.

.. list-table::
   :header-rows: 1

   * - Case directory
     - F1 / F2 WDW
     - F1 LPfr and F2 PKNL
   * - ``em-f1``
     - 1 / 0
     - Off
   * - ``em-f2``
     - 0 / 1
     - Off
   * - ``sine-f1``
     - 3 / 0
     - Off
   * - ``sine-f2``
     - 0 / 3
     - Off
   * - ``em-sine-lp-pknl``
     - 1 / 3
     - On
   * - ``sine-em-lp-pknl``
     - 3 / 1
     - On

Use ``SSB=0`` in F1 and ``SSB=2`` in F2 for the isolated windows,
and ``SSB=2.5`` in F1 and ``SSB=3`` in F2 for the combined recipes.
Use ``SI=256`` in F1 and ``SI=1024`` in F2. For the isolated cases,
zero the manual phases and use ``FCOR=1`` on both axes. For the combined
cases, retain the original prediction lengths, orders and phases, and
use ``FCOR=0.5`` in F1 and ``FCOR=1`` in F2. Run
``pytest tests/qualification/test_topspin_examdata.py -k xfb_windows``.

For the baseline comparisons, set ``NMR_TOPSPIN_XFB_BASELINE_REFERENCES``
to the root containing the following reference groups. Each case
subdirectory contains processed experiment copies ``20``, ``30`` and
``40``. Use ``SI=256`` in F1 and ``SI=1024`` in F2.

- ``bc-modes/bc{1,2,3,4}-f{1,2}`` sets the indicated mode on the
  selected axis. Set all other baseline, window, prediction, phase and
  PKNL settings to zero, and both FCOR values to one.
- ``bc-offset/bc4-f{1,2}-offset`` repeats the isolated mode 4 cases
  with ``COROFFS=314`` Hz in F1 or ``COROFFS=-217`` Hz in F2.
- ``bc-combined/bc{1,2,3,4}-combined`` combines the indicated mode
  in F1 with mode 4 in F2 and zero offsets. Use squared sine in F1,
  an exponential with ``LB=47`` Hz in F2, and ``FCOR=0.5`` in F1.
  Retain the original F1 prediction settings and phases and enable F2 PKNL.

Run ``pytest tests/qualification/test_topspin_examdata.py -k xfb_baseline``.

For Gaussian qualification, set ``NMR_TOPSPIN_XFB_GAUSSIAN_REFERENCES``
to the same reference root. Add the isolated ``bc-modes/bc{5,6}-f{1,2}``
cases with ``BCFW=0.2`` ppm and zero offsets. Add the following cases
under ``bc-gaussian-combined``, with the same windows, prediction,
phases and PKNL as the combined baseline recipes.

- ``sfil-combined`` uses mode 5 and 6, with widths 0.65 and 1.1 ppm
  and offsets 0 and −217 Hz in F1 and F2, respectively.
- ``qfil-combined`` uses mode 6 on both axes, the same widths, and
  offsets 314 and −217 Hz.
- ``qfil-radius-one`` uses the same modes and offsets, with widths
  200 and 20 ppm to exercise the minimum radius.

Run ``pytest tests/qualification/test_topspin_examdata.py -k xfb_gaussian``.

For the fractional-delay comparisons, set
``NMR_TOPSPIN_XFB_FRACTIONAL_REFERENCES`` to the directory of processed
copies. In each case, label ``exam_CMCse_3/16``, ``exam_CMCse_3/19``,
and ``exam_CMCse_1/3`` as ``20``, ``30``, and ``40``, respectively.
Use SI 512 × 4096 (F1 × F2), with windows, phase, baseline, prediction and
PKNL disabled in the ``plain`` case. Use FCOR=1 on both axes.
Enable just F2 PKNL in ``phase``, just F2 WDW=1 with LB=47 Hz in
``em``, and just F2 WDW=3 with SSB=2 in ``sine``. In ``qpol``
and ``qfil``, enable F2 BC_mod=4 or 6, respectively, with COROFFS=−115 Hz
and BCFW=0.7 ppm for the filter. The ``combined`` case uses F1 LPfr,
F1 WDW=4 with SSB=2.5 and FCOR=0.5, and F2 WDW=3 with SSB=3,
qfil with the above settings, and PKNL. Retain the original manual phases
in the combined case. The int32 reference intensities include the
recorded acquisition factor ``2**NC``.

Run ``pytest tests/qualification/test_topspin_examdata.py -k xfb_fractional``.
