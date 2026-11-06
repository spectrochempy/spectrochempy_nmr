Agilent 1D interop qualification
================================

This **Unreleased, experimental** bridge uses the external ``nmrglue-ng``
provider through ``ReadResult → normalize_agilent_fid → from_interop``.
It does not replace public Agilent reader dispatch. No core changes or new
runtime dependencies are needed for this 1D path, including on core 1.1.4.

The qualification pins provider commit
``27df3405b087297c4a347cd3f1c415262078b8e1`` in
``tests/qualification/requirements.txt``. The CI job installs both packages
as wheels, verifies the provider's VCS provenance and imports outside the
checkout, then runs the real-provider tests on Python 3.11 and 3.13.
The usual synthetic facade tests remain separate.

Real acquisitions and provenance
--------------------------------

* The public `nmrML hop example <https://nmrml.org/examples/1/>`_,
  ``FAM013_AHTM.PROTON_04.fid``, contains 32768 complex points, spectral width
  12019.2307692 Hz and observed frequency 599.8311617 MHz. The source describes
  a proton experiment on a Varian VNMRS600 from hop metabolite profiling
  (Farag et al., Metabolomics 8, 492–507, 2012). CI retrieves the original
  archive from that page and verifies archive and member SHA-256 checksums.
* The existing local historical ``agilent_1d`` acquisition contains 1500
  complex points, spectral width 50000 Hz and observed frequency
  125.6811107 MHz. Its redistribution provenance remains unresolved; it is
  tested only when explicitly supplied locally. It is not fetched in CI.

The hashes of both acquisitions are fixed in
``tests/qualification/test_agilent_real.py``. The archive hash and download
URL are fixed in ``tools/fetch_agilent_qualification.py``. No acquisition
binaries are bundled in this repository, wheel or uploaded CI artifacts.
Public availability of the nmrML archive is not a redistribution licence;
this qualification makes no licence claim for its acquisition files.

What is checked
---------------

For each acquisition, a small independent binary decoder checks the actual
file header and interleaved values without either reader library. Raw,
normalized, converted and facade samples must match it exactly, including
complex64 dtype. Time coordinates are checked with relative tolerance
``1e-10`` and absolute tolerance ``1e-14`` seconds, allowing core coordinate
linearization at floating-point precision. The bridge requests unrounded
coordinates with 15 significant digits so short dwell intervals remain
distinct when the core reconstructs coordinates on long acquisitions.
Tests also check dwell, observed frequency,
uninterpreted reference metadata, operation history, ownership, unchanged
source files and dataset persistence. A comparison with the historical
reader checks samples and time coordinates.

Compatible stacking uses two explicitly identified copies of the same
acquisition; it does not establish coverage of two independent compatible
experiments. When the historical corpus is supplied, stacking the two real
acquisitions is refused because they have different sizes and spectral widths.

The profile is storage-normalized: no scan scaling, conjugation, FFT or phase
correction is applied. Both original procpar files lack ``ns`` and ``ne``,
so the historical reader's fallback divisor is one and sample parity is
expected. In particular, the hop file's ``nt=160`` is preserved in raw
metadata but is not mapped to ``NS`` by this provider. Sample parity is not
evidence of correct physical amplitude normalization.

A controlled copy with added ``ns=4`` and ``ne=2`` demonstrates that the
historical reader divides samples by eight while interop preserves them.
This is a metadata variant of the hop acquisition, not another real experiment.

Observed nucleus and remaining scope
------------------------------------

The pinned provider includes the correction from nmrglue-ng PR #153: both the
acquisition nucleus and time-axis label use the observed transmitter ``tn``,
without falling back to the decoupler ``dn``. Missing or empty ``tn`` remains
unknown. For the hop proton acquisition, ``tn=H1`` and ``dn=C13`` now produce
``H1``. An ordinary passing regression checks both exported fields and retains
both original procpar values.

This qualification establishes sample and time-axis preservation on two real
1D acquisitions and correct observed-nucleus selection on the hop file. It
does not establish complete scientific metadata correctness or readiness to
replace the public reader. Resolve the meaning of scan-count/reference fields
before considering dispatch migration. It does not qualify 2D data or calibrated
frequency-domain axes.

Reproduce locally
-----------------

In a dedicated environment with the plugin wheel and core 1.1.4 installed::

   python -m pip install -r tests/qualification/requirements.txt
   python tools/fetch_agilent_qualification.py /tmp/agilent-real
   NMR_AGILENT_REAL_DATA=/tmp/agilent-real \
     python -I -m pytest -ra tests/qualification/test_agilent_real.py

To add the existing historical corpus, also set
``NMR_AGILENT_LEGACY_DATA`` to the directory containing ``agilent_1d``.
Without ``NMR_AGILENT_REAL_DATA`` the suite skips explicitly. Once configured,
missing files, wrong checksums or an unavailable provider fail the suite.
No test modifies the original acquisition files.
