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
