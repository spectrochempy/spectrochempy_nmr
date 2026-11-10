Agilent interop qualification
=============================

The experimental ``read_agilent_interop()`` facade and the qualified
public ``read_agilent()`` routes use the external ``nmrglue-ng``
provider and convert normalized acquisitions with ``from_interop()``.
The 1D qualification includes SpectroChemPy core 1.1.4.

The qualification pins the published nmrglue-ng 1.2.4 wheel in
``tests/qualification/requirements.txt``. The CI job installs both packages
as wheels, verifies the provider version and imports outside the
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
expected. ``NS``/``NE`` remain literal vendor ``ns``/``ne`` fields and are
not replaced by requested or acquired counts. Sample parity is not evidence
of correct physical amplitude normalization.

The pinned provider includes nmrglue-ng PR #154. Its
``dataset.meta.acquisition['scan_counts']`` keeps requested ``nt``, informational
procpar ``ct``, binary block ``ctcount`` and scale evidence separate. Both
original acquisitions agree at 160 (hop) and 512 (historical carbon) scans,
with block scale zero and status ``consistent``. Tests independently read the
binary header and verify these values, their source and status through the
facade, stacking metadata and SCP save/reload. Serialized tuples may become
lists; their values remain identical.

The provider leaves the convenience completed count unknown when evidence is
conflicting, invalid, missing or ambiguous. It never substitutes requested
scans for acquired scans. The plugin preserves this contract without using
the count to scale samples. These real-file tests cover consistent counts
only; stopped acquisitions and invalid/conflicting counters have synthetic
provider coverage, not additional real-acquisition validation here.

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
1D acquisitions, sourced scan-count preservation and correct observed-nucleus
selection on the hop file. It
does not establish complete scientific metadata correctness or readiness to
replace the public reader. Physical amplitude normalization, receiver-gain and
nonzero binary-scale corrections remain unqualified. Explicit frequency
referencing, ppm axes and their source ambiguities are qualified separately
in :doc:`agilent_frequency_reference`. Neither qualification resolves the
conflicting chemical-reference evidence or covers 2D processing.

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

Public single-trace FID reader
------------------------------

With a provider supporting the legacy-presentation contract, the public
``read_agilent()`` reader returns the converted interop dataset for calibrated
single-trace complex FIDs. The provider applies the historical NS/NE divisor
and leaves the completed scan counts as provenance. The direct public FFT
preserves the historical conjugated-input display. An explicit physical
reference through ``Experiment.process(reference=...)`` selects the canonical
FFT convention. Other profiles still use the compatibility reader.


Agilent States 2D public reader
-------------------------------

With a provider that supports States storage normalization,
``read_agilent()`` uses the dataset returned by ``from_interop()`` for
the supported ``phase=[1, 2]`` layout. The provider pairs adjacent
complex FIDs and applies the historical NS/NE divisor once. The plugin
preserves the public branch labels, display coordinates and direct FFT.
The indirect FFT remains guarded until the States processing convention
has been qualified. Profiles outside this layout use the nmrglue-ng compatibility reader.

The historical ``agilent_2d`` dataset is qualified locally. It contains
332 stored traces paired into 166 increments with 1500 complex points
each. The public reader retains the F1 point index because the file has
no ``sw2``. The physical width from ``sw1`` remains available through
the normalized acquisition metadata. Synthetic binary input in CI
checks branch order, a nontrivial divisor and public dataset identity.


Agilent TPPI 2D public reader
-----------------------------

For ``phase=[3]`` with one stored trace per indirect increment,
``read_agilent()`` returns the normalized interop matrix with its
historical public presentation. The real reference has 600 increments
and 1400 complex points per trace. The physical F1 time grid derives
from ``sw1``. The public display retains point indices when ``sw2`` is
absent. The direct FFT remains available. The indirect TPPI FFT is
guarded pending processing qualification. Other layouts use the compatibility reader.


Agilent flat FID series
-----------------------

The qualified series reader preserves complex FIDs in block and trace
order. The normalized contract describes its leading dimension as a
trace index. ``read_agilent()`` retains the historical coordinate
presentation and NS/NE scaling. Profiles outside the normalized contracts use the nmrglue-ng Varian
compatibility reader. It retains all traces in file order when the
declared multidimensional shape disagrees with the binary matrix.

The real OpenVnmrJ CRAFT proton array has four blocks with one trace
each. The separate synthetic test covers two blocks with three traces
each, including a non-unit scale and asymmetric samples.
