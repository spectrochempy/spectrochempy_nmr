nmrML exchange-format qualification
===================================

nmrML is an exchange format. Install nmrglue-ng and ``xmltodict`` to read it
and pass normalized results to ``from_interop()``. For supported 1D acquisitions,
``Experiment`` can also perform the FFT.

Processed spectra
-----------------

The qualification tests use the nmrglue-ng 1.2.4 wheel and ``xmltodict``::

    import nmrglue as ng
    from spectrochempy_nmr import from_interop
    from spectrochempy_nmr.experiment import Experiment

    described = ng.interop.read_nmrml_1d(
        "spectrum.nmrML", read_processed=True, axis_endpoint="last-point"
    )
    source = from_interop(ng.interop.normalize_nmrml_1d(described))
    spectrum = Experiment(source).process()

``axis_endpoint`` is an explicit producer convention: ``last-point`` includes
endValue (N-1 intervals); ``outer-edge`` excludes it (N intervals). The nmrML
schema does not resolve this ambiguity. A ppm axis remains ppm unless the caller
supplies a justified ``observation_frequency_mhz``. The provider never substitutes
acquisition excitation frequency for the processing reference. A supplied MHz
reference is recorded as caller evidence with reference source ``unknown``;
conversion does not certify chemical calibration.

Frequency-domain inputs are not Fourier transformed again. Real data remain
real; manual phase requires an imaginary component. Normalized profiles and
source origin are opaque provenance, not processing selectors. Data, explicit
coordinates, metadata and processing records survive SCP persistence.

Raw complex acquisitions and writer
-----------------------------------

Raw profiles require declared uniform sampling and positive Hz sweep width.
``raw_point_count`` distinguishes complex samples from scalar words, while
``normalize_nmrml_1d(..., conjugation="none" or "conjugate")`` explicitly
chooses sign convention. Time is i/SW seconds. These options need producer or
reference evidence; no digital-filter, phase, gain or reference correction is
implicit. The generic Experiment FFT consumes the selected complex FID.

The provider's existing ``ng.nmrml.read`` contract is extended and
``ng.nmrml.write`` replaces a selected raw/processed array in copied metadata.
It preserves float32/64 and complex64/128 precision, little-endian bytes,
compression choice and other document sections. Counts must match; metadata
are not generated or silently repaired. Optional writer ``schema=`` validation
uses a caller-supplied local XSD and optional lxml before opening the output.
Structural XSD success does not establish scientific correctness.

Qualification and limits
------------------------

Synthetic tests cover raw complex sign/FFT peak position, processed real/complex
samples, explicit ppm/Hz coordinates, no repeated FFT, opaque origins and SCP
round trips. Source bytes and metadata remain unchanged.

Two official public examples are checksum-pinned external inputs, never bundled
or uploaded as artifacts. The hop XML declares Complex128, but an explicit
``data_dtype="<c8"`` produces all 32768 complex samples exactly equal to the
native Varian FID. The default strict profile refuses its inconsistent count;
no dtype guess is introduced. Its excitation field is also inconsistent with
native procpar, so it is not used as a calibrated reference. The Bayesil
processed spectrum has 32768 complex128 samples, verified by independent binary
decoding. Its unrelated FID length is incorrect. Endpoint interpretation remains
conditional on the explicit caller convention. Both old documents fail the
published XSD; their binary round trips are not claimed as schema-valid repair.

For local qualification, fetch into a disposable directory with
``python tools/fetch_nmrml_qualification.py DIRECTORY`` and set
``NMRML_QUALIFICATION_DATA=DIRECTORY``. Missing XML support or reference data cause skips;
a skipped test is not qualification. The dedicated CI path installs the pinned
provider and XML dependency, verifies provenance and downloads checksum-pinned
originals. Public access supports local validation, not a redistribution claim.

Raw 2D echo/anti-echo
---------------------

With nmrglue-ng 1.2.1 and ``xmltodict``, the official HSQC example can be
read in storage order, grouped into adjacent pairs, and imported as an
acquisition::

    import nmrglue as ng
    from spectrochempy_nmr import from_interop

    stored = ng.interop.read_nmrml_raw_2d(
        "bmse000400-exp06.nmrML",
        direct_point_count="scalar",
        official_hsqc_compatibility=True,
    )
    normalized = ng.interop.normalize_nmrml_raw_2d(
        stored, indirect_dwell_s=132.6e-6, confirm_adjacent_pairs=True
    )
    source = from_interop(normalized)

The point-count convention and compatibility option are specific to this
document. The pulse program advances the indirect evolution after each pair.
It contains two evolution delays, giving a dwell of twice the 66.3 µs increment
recorded in the acquisition status. Use evidence from the acquisition when
choosing these settings for another file.

The dataset has dimensions ``(c, y, x)``. The component coordinate labels the
first and second stored branches, and the other coordinates give F1 and F2
times in seconds. Import preserves the real and imaginary samples, their signs,
and their order.

For this acquisition, explicitly select Bruker Rance–Kay reconstruction::

    from spectrochempy_nmr.experiment import Experiment

    spectrum = Experiment(source).process(echo_antiecho="bruker-rance-kay")

The reconstruction combines the first and second stored traces into
``first - second`` and ``1j * (first + second)`` before the direct FFT.
It uses the original trace lengths for both FFTs and returns the four planes
``rr``, ``ri``, ``ir``, and ``ii`` with descending carrier-relative Hz axes.
The profile applies no additional F1 sign alternation. On this HSQC, an
additional alternation moves the two dominant peaks by half the F1 sweep
width compared with the native Bruker processed spectrum. The order of
stored traces alone cannot determine the reconstruction for another pulse
program.

When the imported result carries Bruker acquisition parameters, choose the
method and let ``Experiment`` resolve the delay::

    spectrum = Experiment(source).process(
        echo_antiecho="bruker-rance-kay", digital_filter="phase"
    )

The delay comes from normalized acquisition metadata, ``GRPDLY`` in
``acqus``, or a known legacy ``DSPFVS`` / ``DECIM`` table entry. For the
official HSQC nmrML file, the parameters are in the accompanying Bruker
``acqus``. Its ``DSPFVS=12`` and ``DECIM=24`` give a delay of 70⅙ complex F2
points. If only the XML was imported, supply the delay explicitly::

    spectrum = Experiment(source).process(
        echo_antiecho="bruker-rance-kay",
        digital_filter="phase",
        bruker_group_delay=70 + 1 / 6,
    )

``"phase"`` applies the NMRPipe ``FT -df`` phase ramp before the F1
transform and preserves all 2048 F2 bins. ``"time"`` corrects both traces
using the historical Bruker reader algorithm before reconstruction and
returns 1976 F2 bins for this HSQC. The methods handle the trace boundaries
differently. Omit the filter option to process the stored traces as-is.
The processing trace records the method, delay, its source, and the number
of removed points. Correction requires raw traces that still contain the
delay. Apodization, zero-filling, subsequent phase correction and chemical
referencing remain separate steps.

For local qualification, download the
`official HSQC XML <https://nmrml.org/examples/6/bmse000400-exp06.nmrML>`_
and extract ``ser`` from the
`matching BMRB archive <https://bmrb.io/metabolomics/mol_summary/zip_entry_directory.php?experiment=6&id=bmse000400>`_.
Set ``NMRML_HSQC_XML`` and ``NMRML_HSQC_SER`` to the downloaded file paths
and run ``pytest tests/qualification/test_nmrml_interop.py -k official_hsqc``.
The test verifies both file hashes, compares both branches against the native
SER, and verifies SCP persistence.

To qualify processing, also extract ``pdata/1/2rr``, ``pdata/1/procs``,
and ``pdata/1/proc2s`` from that archive. Generate the NMRPipe checkpoints
locally with ``tools/nmrpipe/regenerate_echo_antiecho_oracle.py`` and set
``NMRML_HSQC_ORACLE`` to the resulting NPZ and ``NMRML_HSQC_PDATA`` to the
extracted ``pdata/1`` directory. Run
``pytest tests/test_nmrpipe_echo_antiecho_oracle.py`` to check reconstruction,
both FFTs, axes, and the F1 placement of the two main native peaks.

To also qualify the F2 group-delay correction, generate another oracle with
``--digital-filter`` and set ``NMRML_HSQC_DMX_ORACLE`` to its path.
Extract ``acqus`` from the BMRB archive and set ``NMRML_HSQC_ACQUS`` to
its path to verify automatic delay resolution and the time-domain correction.

The HSQC qualification also compares both F2 branches against the released
nmrglue-ng wheel using its post-FFT correction, with the
`phase-sign adaptation documented in nmrglue-ng #194 <https://github.com/spectrochempy/nmrglue-ng/pull/194>`_.
