Interoperability and referencing
================================

Convert normalized nmrglue-ng results to SpectroChemPy datasets and
work with NMR frequency references. For exchange files, start with
the :ref:`public reader <reading-exchange-formats-through-interop>`.

Raw 2D component contract
-------------------------

For qualified QF, States, States-TPPI, TPPI and echo/anti-echo inputs,
``read()`` and ``from_interop()`` arrange the declared components and retain
the physical axes and acquisition widths. Explicit component labels determine
the ordering: permutations of the same complete label set produce the same
native array. A selected States input convention is converted at this boundary,
with the operation recorded in the dataset metadata.

``Experiment(dataset).process()`` selects the Fourier reconstruction from
quadrature and component meaning. It does not select an algorithm from the
vendor, file profile or experiment name. A saved SCP dataset retains this
information and can be processed without reopening the original file.
Echo/anti-echo inputs use the normalized Rance–Kay arrangement by default;
``echo_antiecho="rance-kay"`` is optional, and the historical
``"bruker-rance-kay"`` spelling remains an alias. Digital-filter correction
uses the imported delay descriptor or an explicit caller override, rather
than rereading vendor parameters during this common processing path.

An undetermined physical quadrature remains unsupported for a complete
indirect transform. Importing stored components does not establish that
quadrature. This contract does not qualify additional file formats or modes.
Recorded TopSpin replay and specialized experimental helpers remain separate
entry points.

JCAMP-DX FID pages and QF processing
------------------------------------

With nmrglue-ng 1.4.0 or later providing ``jcampdx-fid-2d``, use the explicit
interop reader for regular ``T1,T2,R,I`` FID pages. Confirm QF from the
acquisition convention; the declared scheme must also be non-phase-sensitive.
The complex-sign choice is explicit. For a TopSpin export whose stored
complex samples are to be retained:

.. code-block:: python

   import spectrochempy as scp

   raw = scp.nmr.read(
       "qf_fid.dx", protocol="interop", profile="jcampdx-fid-2d",
       normalize_options={"quadrature": "qf", "conjugation": "none"},
   )
   spectrum = scp.nmr.Experiment(raw).process()

The existing generic QF processor consumes the normalized components and
time grids; it does not recognize COSY or HMBC names. Results have relative
frequency coordinates in Hz. Reading does not remove the digital filter or
establish a chemical-shift reference, and this FFT-only example does not
apply phasing, apodization or magnitude. Inspect any recorded group delay
and qualify its correction before interpreting phases or intensities.
Saving the raw dataset as SCP retains its processing contract.

Echo/anti-echo processing
~~~~~~~~~~~~~~~~~~~~~~~~~

nmrglue-ng 1.5.0 also supports declared echo/anti-echo FIDs with direct
simultaneous DQD acquisition. Confirm that each adjacent trace pair belongs
to one physical indirect increment, and establish the time between pairs
from acquisition provenance. Do not infer it by simply subsampling the stored
T1 row coordinates. The reader preserves the pair as two complex channels;
``Experiment.process()`` performs the existing Rance-Kay reconstruction.

For the qualified EPFL HSQC acquisition, the indirect spectral width checked
against TopSpin import is 16608.16082508 Hz. For another acquisition, replace
this value with its established width:

.. code-block:: python

   indirect_width_hz = 16608.16082508
   raw = scp.nmr.read(
       "echo_antiecho_fid.dx", protocol="interop", profile="jcampdx-fid-2d",
       normalize_options={
           "quadrature": "echo-antiecho", "conjugation": "none",
           "confirm_adjacent_pairs": True,
           "indirect_dwell_s": 1.0 / indirect_width_hz,
       },
   )
   spectrum = scp.nmr.Experiment(raw).process()

The result has four signed component planes (rr, ri, ir, ii), with relative
Hz axes. No filter correction or chemical-shift referencing is implicit.
If an unapplied positive group delay is recorded and its meaning has been
established, phase-domain correction can be requested explicitly:

.. code-block:: python

   corrected = scp.nmr.Experiment(raw).process(digital_filter="phase")

The EPFL reference records a delay of 76 complex direct points. Four TopSpin
compression modes give identical normalized data and processed planes.
The planes were compared to NMRPipe with and without phase-domain correction;
this qualifies the stated operations, not an absolute experimental phase or
ppm reference. Time-domain correction is also checked against the native
Bruker route. Saved SCP data retains the common processing contract, without
requiring the source reader or original vendor metadata during processing.
States and States-TPPI processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nmrglue-ng 1.5.0 also normalizes declared States and States-TPPI DQD
FID pages. Use the matching ``quadrature`` and ``conversion_profile``;
confirm adjacent pairs and provide the physical dwell from acquisition
provenance, just as for echo/anti-echo. Four stored component planes are
adapted into E1/E2 complex channels. States-TPPI alternation is applied by
processing, not by reading.

For an acquisition whose established indirect width is ``indirect_width_hz``:

.. code-block:: python

   raw = scp.nmr.read(
       "states_tppi.dx", protocol="interop", profile="jcampdx-fid-2d",
       conversion_profile="states-tppi",
       normalize_options={
           "quadrature": "states-tppi", "conjugation": "none",
           "confirm_adjacent_pairs": True,
           "indirect_dwell_s": 1.0 / indirect_width_hz,
       },
   )
   spectrum = scp.nmr.Experiment(raw).process()

For pure States, use ``"states"`` in both options. The same common FFT engine
handles the imported channels without vendor or experiment-name dispatch.
SCP round trips preserve this contract. Results have four signed planes and
relative Hz axes; absolute phase and ppm references remain unqualified.

Explicit ppm references after the common 2D FFT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Normalized QF, States, States-TPPI and echo/anti-echo FIDs can be calibrated
with explicit point anchors after their common FFT:

.. code-block:: python

   spectrum = scp.nmr.Experiment(raw).process(reference={
       "F1": {"reference_frequency_mhz": 100.0, "point_index": 0,
              "ppm": 12.0, "source": "processing"},
       "F2": {"reference_frequency_mhz": 400.0, "point_index": 0,
              "ppm": 10.0, "source": "processing"},
   })

Replace these illustrative values with known references. The zero-based
indices refer to the output FFT grid, including any requested QF zero-fill.
For each axis, ``ppm[j] = ppm_anchor + (Hz[j] - Hz_anchor) / MHz``.
The supplied reference frequency is distinct from the acquisition observation
frequency. ``source`` defaults to ``unknown``. Intensities, signed components
and sample order are unchanged; axes and reference provenance survive SCP
save/reload. Without ``reference``, the existing Hz grids are preserved.
Calibration does not establish phases or reproduce a full vendor recipe.

Explicit processing recipes for raw JCAMP data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For QF, States, States-TPPI and echo/anti-echo inputs, ``replay`` also accepts a
serialized ``topspin-xfb-v1`` plan captured from a matching native read:

.. code-block:: python

   native = scp.read_topspin(
       "native/ser", local_only=True,
       remove_digital_filter=False, remove_dc_offset=False,
   )
   plan = native.meta.nmr_processing["vendor_profile"]["xfb_plan"]
   spectrum = scp.nmr.Experiment(raw).process(replay=plan)

The plan must exist and describe the intended acquisition and recipe. It
contains the supported windows, sizes, digital-filter delay, phases and ppm
calibration. The processor verifies the physical quadrature, original shape,
spectral widths (relative tolerance ``1e-12`` for serialization roundoff)
and uncorrected uniform time grids. Both input and recipe widths are recorded.
An explicit QF ``normalize_options={"quadrature": "qf", "conjugation": "none",
"indirect_dwell_s": physical_dwell}`` can establish a known acquisition T1
step when the JCAMP grid differs; this requires nmrglue-ng 1.5.0.
A genuine width mismatch is rejected. It rejects incompatible
plans and individual processing overrides. It interprets no raw vendor
parameters from JCAMP metadata. The serialized plan can be kept separately
and reused after saving/reloading the raw dataset as SCP.

JCAMP import applies its declared YFACTOR; recipe execution does not infer
scan/gain normalization from missing acquisition parameters. Compare data on
the same declared intensity scale. Passing a plan does not broaden the supported XFB operations. States uses
the States-family signed-plane arrangement without TPPI alternation.
Capture requires a complete native status and a non-null plan; a legacy
read with truncated points or unsupported processing status is insufficient.

Processed real 2D exports
~~~~~~~~~~~~~~~~~~~~~~~~~

nmrglue-ng 1.5.0 accepts native TopSpin real-plane NTUPLES with a
per-page dependent dimension. Import them as frequency-domain data:

.. code-block:: python

   spectrum = scp.nmr.read(
       "processed.dx", protocol="interop", profile="jcampdx-spectrum-2d",
   )

The declared F1/F2 coordinates and YFACTOR-scaled exported intensities are
preserved. Do not apply another FFT. Native TopSpin 4.5.0 RSPEC+ISPEC examples
export only the real ``2rr`` plane, with no imaginary pages. This route does
not reconstruct 2ri/2ir/2ii or apply additional NC_proc or scan/gain scaling.
Its axes remain the declared Hz or ppm grids; no chemical zero is inferred.

nmrglue-ng 1.5.0 can explicitly calibrate both declared Hz axes to ppm:

.. code-block:: python

   spectrum = scp.nmr.read(
       "processed.dx", protocol="interop", profile="jcampdx-spectrum-2d",
       normalize_options={"axis_reference": {
           "F1": {"reference_frequency_mhz": 100.0, "point_index": 0,
                  "ppm": 12.0, "source": "processing"},
           "F2": {"reference_frequency_mhz": 400.0, "point_index": 0,
                  "ppm": 10.0, "source": "processing"},
       }},
   )

Replace these illustrative values with a known zero-based point and reference
frequency for each axis. For native Bruker processed grids, use ``proc2s``
(F1) and ``procs`` (F2): ``SF`` in MHz, index zero and ``OFFSET`` in ppm.
Processing ``SF`` can differ from acquisition ``SFO1``. The affine conversion
preserves intensities and sample order. Coordinate ``reference_frequency``
retains the supplied MHz value; it is also called ``acquisition_frequency``
only when explicitly declared as acquisition provenance. Ppm coordinates survive ``.scp``
save/reload. Both axes must initially be in Hz. No chemical origin or vendor
parameter-to-axis mapping is inferred. This option does not reference raw
FID Fourier transforms.

.. currentmodule:: spectrochempy_nmr

.. autosummary::
   :nosignatures:
   :toctree: generated

   from_interop
   set_nmr_context

Export processed spectra
-------------------------

See :doc:`/userguide/exchanging_spectra` for the supported components,
axis units and round-trip examples.

.. autosummary::
   :nosignatures:
   :toctree: generated

   write_jcamp
   write_csdf

Experimental entry points
-------------------------

.. autosummary::
   :nosignatures:
   :toctree: generated

   read_agilent_interop
   reference_agilent_1d
   reference_interop_1d


TPPI and QSEQ JCAMP processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Development nmrglue-ng builds support declared TPPI and QSEQ JCAMP FIDs with
direct DQD acquisition. Supply the physical dwell between successive F1
samples from acquisition provenance. Do not infer it from the exported T1
interval. QSEQ sign alternation is applied once by normalization, producing
the common TPPI representation; the original scheme remains in metadata.

For a validated acquisition whose physical indirect dwell is known:

.. code-block:: python

   raw = scp.nmr.read(
       "qseq.dx", protocol="interop", profile="jcampdx-fid-2d",
       normalize_options={
           "quadrature": "qseq", "conjugation": "conjugate",
           "indirect_dwell_s": physical_indirect_dwell_s,
       },
   )
   spectrum = scp.nmr.Experiment(raw).process()

This explicit conjugation matches the controlled TopSpin convention; establish
the sign convention for another acquisition. A declared filter descriptor
must record an unapplied, explicitly known zero group delay for this FFT
path by default. Unknown or nonzero delays are rejected. For a known finite,
unapplied delay shorter than F2, ``process(digital_filter="none")`` explicitly
retains the delay without correction and records this choice. This diagnostic
FFT does not remove the filter or provide an absorptive spectrum.
``process(digital_filter="phase")`` instead compensates the known delay on the
shifted F2 frequency grid, before the real F1 transform. It keeps all F2 points
and records the correction as applied; missing, invalid or already-applied
filter descriptors are rejected. No delay override is inferred. Windows,
manual phasing, variable FFT sizes, time-domain filter trimming and vendor-recipe
replay remain outside this path.
The default output consists of four signed planes ``rr, ri, ir, ii`` on
relative Hz grids. ``process(reference={"F1": ..., "F2": ...})`` accepts the
same explicit point anchors documented above, without changing components.
Choose either point anchors or ``tppi_reference``, not both.

Saved SCP data preserves this processing contract without the provider or
original files. Controlled synthetic TopSpin references and independent matrix
DFTs check the numerical transforms. The public native TPPI qualification below
also checks uncorrected and phase-compensated FFTs on experimental acquisitions.
Native QSEQ and absolute experimental phases remain unqualified. TopSpin
presentation uses ``RR, -IR, -RI, II`` and replaces the central ``2rr`` bin
with its two F1 neighbors' average. The common FFT preserves that bin rather
than applying this vendor display operation.

The acquisition-mode definitions are documented in the Bruker
`Acquisition Reference <https://nmrcore.chem.kuleuven.be/media/files/guides/bruker_manuals_4.2.0/acquisition-reference.pdf>`_
under ``FnMODE``. The Bruker
`Processing Reference <https://nmr.chem.ucsb.edu/docs/Bruker_NMR_Manuals/processing-reference_v007.pdf>`_
distinguishes the real F1 transforms used by TPPI and QSEQ.

Public experimental TPPI qualification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Young lab's `2025-indole-azo-sterics dataset
<https://github.com/the-young-lab/2025-indole-azo-sterics/tree/73af741bff1059a2a8491254a5f765644b985952>`_
is CC0. Under ``Data for SI/1H and 2D NOESY NMR``, the acquisitions
``OMe-PAI-H cis NOESY`` and ``OMe-PAI-H trans NOESY/1`` explicitly record
``acqu2s.FnMODE=3`` (pure TPPI), ``acqus.AQ_mod=3`` (DQD), and
``GRPDLY=76``. Each raw matrix has 256 F1 samples and 1024 complex F2 samples.
These are original experimental acquisitions, not relabelled States-TPPI data.

Validation uses copies processed with TopSpin 4.5.0: F1/F2 ``SI=256/1024``,
``TDeff=256/2048``, ``WDW=BC_mod=ME_mod=PHC0=PHC1=TDoff=STSR=0``,
``FCOR=1``, ``PKNL=no``, ``REVERSE=no``, ``STSI=SI`` and ``xfb y nodisp``.
Keep the native acquisition parameters, including filter metadata, unchanged.
This produces a reference with no windows, phase or filter correction.
Do not compare this FFT to the study's original phased/windowed spectra.

Export each copy's raw data as FIX, PACKED, SQUEEZED and DIFF/DUP, and its
processed real plane as DIFF/DUP. With a Bruker current-data parameter file
``curdat`` pointing to that copy, the installed exporter accepts:

.. code-block:: bash

   "$XWINNMRHOME/prog/bin/jcampdx" curdat tojdx sample-FIX.dx 0 0 TPPI Young CC0
   # Raw mode 0; codec index 0/1/2/3 = FIX/PACKED/SQUEEZED/DIFF_DUP.
   "$XWINNMRHOME/prog/bin/jcampdx" curdat tojdx sample-processed.dx 2 3 TPPI Young CC0

Name the other exports ``sample-PACKED.dx``, ``sample-SQUEEZED.dx`` and
``sample-DIFF_DUP.dx``. From a plugin source checkout, run the read-only tool:

.. code-block:: bash

   python tools/qualify_public_tppi.py ORIGINAL COPY/pdata/1 EXPORTS --prefix sample

The tool verifies processing settings, exact native/raw-JCAMP equality for all
four encodings, the common public FFT's four signed planes, both ppm grids,
exact processed-JCAMP/TopSpin stored RR equality, and unchanged input hashes.
It supplies ``1/(2*acqu2s.SW_h)`` as the physical F1 dwell for these Bruker TPPI
references; the JCAMP T1 interval alone is insufficient. It compares absolute
raw-count amplitudes with TopSpin's recorded ``NC_proc`` scale, without fitting
an amplitude or dividing by NS/RG. This does not establish scan/gain-normalized
physical intensities or absolute phase. The central RR presentation operation
is applied only to a comparison copy. QSEQ remains covered by controlled
synthetic references; no native indirect-QSEQ acquisition is claimed here.


TPPI digital-filter phase compensation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same two public TPPI acquisitions are qualified against TopSpin 4.5.0 with
all the settings above, except ``PKNL=yes`` on F2 (F1 remains ``PKNL=no``).
The original ``SER``, ``acqus`` and ``acqu2s`` stay byte-identical, including
``GRPDLY=76``. Keep this corrected reference in a separate copy from the
uncorrected reference. Export its processed RR into a separate export directory;
the four raw JCAMP exports can be reused unchanged.

.. code-block:: python

   # raw is the explicitly normalized TPPI FID from the example above.
   corrected = scp.nmr.Experiment(raw).process(
       digital_filter="phase",
   )

Use the same read-only qualification tool with ``--digital-filter phase``
and the corrected reference/export paths. It verifies F2 ``PKNL=yes``, F1
``PKNL=no``, exact raw decoding and processed RR, unchanged source hashes,
four signed FFT planes and both ppm axes. Relative L2 errors on each plane
are below ``2.81e-7`` for cis and ``2.28e-7`` for trans (threshold ``2e-6``).
There is no fitted scale, shift or phase. Only the TopSpin central RR display
operation is applied to a comparison copy, as for the uncorrected reference.

For F2 FFT size N and stored delay g, the compensation multiplies shifted
complex F2 bin k by ``exp(-2j*pi*g*k/N)``. It precedes F1 because F1 transforms
the real and imaginary F2 sequences separately. No samples are trimmed; default
Hz grids and explicit ppm anchors are preserved. The returned acquisition and
processing trace record an applied phase correction with zero removed points;
the input FID remains unchanged. Saved SCP FIDs can reproduce the calculation
without the provider; saved spectra preserve the applied state.

Independent matrix DFT tests cover zero, integer and fractional delays. Vendor
parity is established for these two native 76-point-delay acquisitions only;
other digital filters, time-domain correction, absolute experimental phase,
scan/gain-normalized physical intensities and full TPPI XFB replay require
separate qualification. This is a group-delay correction, not a proof that the
remaining experimental spectrum is correctly phased.
