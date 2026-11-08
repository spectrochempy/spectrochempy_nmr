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
The public TopSpin reader keeps its historical route for raw 1D data.

Normalization explicitly trims zero padding to TD/2, optionally applies the
existing truncated group-delay correction, and conjugates into the canonical
DFT orientation. It preserves the original samples and raw metadata in the
input result and records the transformations in ``applied_operations``.
The neutral descriptor supplies BF1/O1 acquisition referencing. Its opaque
``acquisition`` candidate must be selected explicitly; no processed-data SF
reference or vendor phase/apodization recipe is replayed. Changing origin,
profile identity or raw metadata does not change Experiment processing.

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
