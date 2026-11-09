Normalized Bruker 1D qualification
==================================

This workflow is **Unreleased and experimental**. It demonstrates that a second
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
