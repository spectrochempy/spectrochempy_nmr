Normalized processed 1D spectra
===============================

This note describes the interop contract for spectra already in the
frequency domain. For ordinary TopSpin imports, use
:doc:`the reader guide <userguide/reading_data>`. For supported Bruker 1D
``pdata`` profiles, ``read_topspin()`` delegates reading and normalization
to nmrglue-ng while preserving the reader's real or complex presentation,
receiver gain, metadata and coordinates.

For qualified Bruker 1D spectra, the public reader returns the dataset built
by ``from_interop()`` using the nmrglue-ng 1.2.1 axis and receiver-gain contract. It keeps the acquisition frequency (``SFO1``) in
coordinate metadata. With ``proc_axis=True``, ppm/Hz conversions use the
processing reference (``SF``), which can differ from ``SFO1``.
The operation trace records receiver gain separately from ``NC_proc``.

``process()`` recognizes the supplied frequency domain and preserves samples
and coordinates. It does not run another FFT, apodization, zero filling or
reference calibration. Explicit time-domain requests are rejected. The returned
copy carries a processing trace; source data and metadata are not modified.

If only ``1r`` was requested or supplied, the spectrum stays real. Additional
phase correction requires the available imaginary component; it never invents
one. For complete complex input, ``process(phase="manual", phc0=...)`` applies
an explicit additional phase without moving the axis. It does not replay phase
parameters from raw vendor metadata. Inverse reconstruction of a source FID,
especially from a real-only spectrum, is outside this qualification.

Export a processed 1D spectrum
------------------------------

Use the NMR plugin writer for a real, unmasked spectrum with intensities
without physical units and a regular Hz or ppm coordinate::

   import spectrochempy as scp

   spectrum = scp.read_jcamp("processed_proton.jdx")
   path = scp.nmr.write_jcamp(spectrum, "export.jdx")
   restored = scp.read_jcamp(path)

Reading NMR through ``scp.read_jcamp()`` requires a core version that
includes the NMR routing introduced after SpectroChemPy 1.2.1.

The writer uses the stored coordinate and preserves its units and direction.
A slice can contain irregular coordinates if the core rounded the axis
during slicing. For finely spaced grids, construct the coordinate with
``rounding=False`` and ``sigdigits=15`` before slicing. The writer
rejects irregular stored grids.

The dataset title must be printable ASCII. If observation metadata are
missing, pass ``observation_frequency`` in MHz (or as a compatible quantity)
and ``nucleus`` as a string such as ``"^1H"``. Use ``overwrite=True`` to
replace an existing local file.

Bruker TopSpin 4.5.0 can import the tested 1D files using ``fromjdx``.
Its imported processed data use integer intensities. The small synthetic
test lost fractional intensity on import, while the tested real Bruker
spectrum preserved its integer values. TopSpin can misread scientific
notation in the data table. The writer uses plain decimals when they fit
and may still use scientific notation for very small or large values.
JCAMP exports from TopSpin can also use a relative Hz axis with a separate
chemical shift reference. Re-export preserves the relative Hz coordinates
and omits the source ``.SHIFTREFERENCE`` record. To retain chemical shifts
in the exported file, supply a spectrum with an explicitly referenced
ppm coordinate.

References and amplitudes
-------------------------

The reference comes from processing SF/OFFSET/SW_p. In Hz,
``axis[i] = OFFSET*SF - i*SW_p/N``; divide by processing SF in MHz for ppm.
Acquisition BF1/SFO1 remain source metadata and do not replace SF. This preserves
an existing processed reference rather than guessing a new chemical standard.
Both frequency-axis directions are accepted by the generic converter; grids
must be finite and monotonic. Integer coordinates must be representable in the
core's floating-point coordinates or are rejected rather than silently rounded.

The provider combines the components supplied by name, with no conjugation or
extra scaling. ``scale_data=True`` applies the reader's NC_proc storage decoding
once; ``False`` preserves unscaled values. Neither choice adds RG/NS amplitude
normalization. The original reader corrections remain in the provenance trace.
The first Bruker profile covers complete SI-matched spectra only; extracted
STSI/STSR grids are rejected until separately qualified.

Real qualification
------------------

Two original datasets are tested: the existing SCP TopSpin spectrum (16384
points, NC_proc=-5) and the public nmrML tomato example (32768 points,
NC_proc=-1). Both contain real and imaginary components. Tests exercise real-only
and complex imports, scaled and unscaled storage, exact component values,
processing axes, no re-FFT, metadata independence and SCP persistence.
Hashes are checked before and after; no original payload or derived spectrum
is bundled or uploaded as an artifact. Public availability does not establish
redistribution rights. The tomato source is the
`nmrML example2 <https://nmrml.org/examples/2/>`_.

Fetch and run the public original separately::

   python tools/fetch_pdata_qualification.py /path/to/external-corpus
   export NMR_PDATA_TOMATO_DATA=/path/to/external-corpus/MMBBI_10M12-CE01-1a/1/pdata/1
   export NMR_PDATA_TOPSPIN_DATA=/path/to/topspin_1d/1/pdata/1
   python -I -m pytest -ra tests/qualification/test_pdata_real.py

The fetcher validates archive/member SHA-256 values and refuses to overwrite
mismatched existing files. CI downloads the pinned public tomato example and
runs its four component/scaling variants. Local TopSpin cases are optional in
CI: unset paths are explicit skips, never reported as real validation.

External comparison
-------------------

``tools/qualify_pdata_nmrpipe.py`` imports each int32 component into NMRPipe as
already-frequency-domain data. It supplies the independently declared last-point
origin with ``-xORIG``; the default real-FID origin model is not a processed
spectrum calibration. It never calls FT or phase processing. Parameters are
explicit oracle inputs, not values taken from the normalizer::

   python tools/qualify_pdata_nmrpipe.py /path/to/topspin_1d/1/pdata/1 \
     --nmrpipe-bin /path/to/nmrpipe/bin --points 16384 --spectral-width 250000 \
     --reference-mhz 500.071553859985 --offset-ppm 253.0909 --nc-proc -5

   python tools/qualify_pdata_nmrpipe.py /path/to/MMBBI_10M12-CE01-1a/1/pdata/1 \
     --nmrpipe-bin /path/to/nmrpipe/bin --points 32768 \
     --spectral-width 6002.40096038416 --reference-mhz 500.159950648881 \
     --offset-ppm 11.09915 --nc-proc -1

NMRPipe 13 comparisons give component errors below 3e-8 relative and maximum
axis error below 9e-6 ppm (tolerance 2e-5 for float32 header coordinates). This
checks preservation of components and explicitly supplied calibration, not an
independent chemical-standard assignment or acquisition-processing recipe.

Varian/Agilent processed data
-----------------------------

Processed Varian data do exist. The public
`nmrML example4 <https://nmrml.org/examples/4/>`_ contains a Varian-origin
biosample spectrum processed by Bayesil, including a 32768-point complex128
spectral array. The existing low-level
``ng.nmrml.read(path, read_processed=True)`` reads this array when the optional
``xmltodict`` dependency is installed. A normalized producer with qualified
frequency-axis metadata is still needed; this increment adds the Bruker pdata
profile only.
No native ``phasefile`` or ``datdir/data`` original was found in the local
corpus searched for this task. Those native processed inputs would need their
own format/reference qualification. The existing Varian FID profile remains
separate from this availability finding.
