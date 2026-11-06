Explicit nmrML 1D exchange-format qualification
===============================================

This optional path consumes nmrglue-ng's normalized 1D results through the
existing generic ``from_interop`` and ``Experiment`` APIs. It adds no runtime
source change, provider dependency or automatic reader dispatch. nmrML is an
exchange format; this does not add native Varian processed reading.

Processed spectra
-----------------

The qualification requirements pin provider commit
``0a7421795a9008c57eb63d6d747debc45ecf7135`` from
`provider PR #160 <https://github.com/spectrochempy/nmrglue-ng/pull/160>`_.
It is an unreleased candidate, not a capability promised by older provider
packages. Install that optional provider and ``xmltodict`` separately::

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
``NMRML_QUALIFICATION_DATA=DIRECTORY``. Missing optional provider/data cause skips;
a skipped test is not qualification. The dedicated CI path installs the pinned
provider and XML dependency, verifies provenance and downloads checksum-pinned
originals. Public access supports local validation, not a redistribution claim.

No multidimensional, nonuniform, XY-pair, native processed Varian, universal
conversion or unqualified chemical-reference claim is made. No package release
or upstream publication follows from this qualification.
