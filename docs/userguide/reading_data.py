# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [raw] raw_mimetype="text/restructuredtext"
# Reading NMR data
# ================
#
# The development version of the Bruker reader leaves digital-filter and
# DC-offset correction to processing by default. They decode the data, represent the acquisition
# quadrature, and calibrate the time coordinates when the format provides
# the required metadata. Bruker FID and SER readers also normalize amplitudes
# by scan count and receiver gain. Use :doc:`processing_1d` or
# :doc:`processing_2d` to select a correction during processing.
#
# ``scp.nmr.read(...)`` dispatches public readers for TopSpin/Bruker, Agilent or
# Varian, JEOL and SIMPSON inputs. A protocol can be supplied explicitly
# when automatic detection is not appropriate.

# %%
import spectrochempy as scp

# %% tags=["skip-execution"]
bruker = scp.nmr.read("path/to/1/fid")
processed = scp.nmr.read("path/to/experiment", expno=1, procno=1)
agilent = scp.nmr.read("path/to/fid", protocol="agilent")

# %% [raw] raw_mimetype="text/restructuredtext"
# Reader catalogue
# ----------------
#
# Use the calls below with your file or experiment directory. The available
# processing steps depend on the acquisition and file layout. See the
# :doc:`../reference/readers` for reader options.
#
# .. list-table::
#    :header-rows: 1
#    :widths: 27 34 39
#
#    * - Format
#      - Recommended call
#      - Typical inputs
#    * - Bruker TopSpin
#      - :py:func:`scp.nmr.read_topspin(path) <spectrochempy_nmr.read_topspin>`
#      - FID, SER, or processed ``pdata`` in an experiment directory
#    * - Agilent/Varian
#      - :py:func:`scp.nmr.read_agilent(path) <spectrochempy_nmr.read_agilent>`
#      - Acquisition directory containing ``fid`` and ``procpar``
#    * - JEOL 1D
#      - :py:func:`scp.nmr.read_jeol(path) <spectrochempy_nmr.read_jeol>`
#      - Supported 1D JDF FIDs
#    * - JEOL proton COSY
#      - ``scp.nmr.process_jeol_cosy_magnitude(path)``
#      - Supported uniform proton COSY JDFs, processed to a 2D magnitude spectrum
#    * - SIMPSON
#      - :py:func:`scp.nmr.read_simpson(path) <spectrochempy_nmr.read_simpson>`
#      - Simulated FIDs and spectra, or a simulation input with a sibling output
#    * - CSDF
#      - ``scp.nmr.read_csdf(path)`` or ``scp.nmr.read(path)``
#      - Self-contained processed 1D or 2D spectra on linear frequency axes
#    * - NMRPipe
#      - :py:func:`scp.nmr.read(path, protocol="interop") <spectrochempy_nmr.read>`
#      - Supported time- and frequency-domain files
#    * - nmrML
#      - :py:func:`scp.nmr.read(path, protocol="interop", profile=...) <spectrochempy_nmr.read>`
#      - Supported acquisition and spectrum profiles
#    * - NMR JCAMP-DX
#      - :py:func:`scp.nmr.read(path, protocol="interop", profile=...) <spectrochempy_nmr.read>`
#      - Supported NMR profiles, such as ``"jcampdx-spectrum-1d"``
#
# See :doc:`jeol_cosy_magnitude` for a complete raw-JDF-to-2D-spectrum
# workflow on uniform proton COSY acquisitions. Other JEOL 2D acquisitions
# (including HSQC, HMQC and HMBC) and NUS acquisitions are not supported.
# See :doc:`exchanging_spectra` for export and round-trip examples.
#
# See :ref:`reading-exchange-formats-through-interop` for the profile and
# convention options for exchange formats.
#
# Read an already processed TopSpin spectrum
# ------------------------------------------
#
# Pass ``procno`` to select the processed spectrum in an experiment. The
# returned dataset already has a frequency axis and can be plotted or
# analyzed directly.

# %%
spectrum = scp.nmr.read_topspin(
    "nmrdata/bruker/tests/nmr/topspin_1d", expno=1, procno=1
)
_ = spectrum.real.plot()

# %% [raw] raw_mimetype="text/restructuredtext"
# Use ``proc_axis=True`` to use the reference saved during TopSpin processing.
# This is useful when the spectrum was re-referenced after acquisition.
#
# Bruker F1 acquisition mode
# --------------------------
#
# With SpectroChemPy 1.2.0 or later, if the indirect spectral width cannot
# be determined, supply the acquisition width in Hz (``SW_h``) or restore
# the indirect ``SW`` and ``SFO1`` parameters before 2D processing.
#
# For a raw Bruker SER, the reader uses the indirect acquisition parameter
# ``FnMODE``. If it is missing or undefined, the reader uses ``MC2`` from
# the indirect processing parameters when available.
#
# Reopen the acquisition with its known mode, for example
# ``scp.nmr.read_topspin(path, quadrature="states-tppi")``, then apply the
# F2 and F1 transforms in order. The accepted choices are ``"qf"``,
# ``"states"``, ``"states-tppi"`` and ``"echo-antiecho"``.
# Older core versions require a supported mode and valid F1 width at import.
# For an undetermined mode with a valid width, supply ``quadrature="qf"``
# if the acquisition used QF.
#
# See :doc:`bruker_indirect_sampling` for complete 2D processing of
# interrupted States-TPPI acquisitions.
#
# Agilent/Varian States quadrature
# --------------------------------
#
# For a paired 2D acquisition with ``phase=[1,2]``, choose the F1 convention
# explicitly when it is known::
#
#    fid = scp.nmr.read_agilent("sample.fid", quadrature="states")
#    spectrum = scp.nmr.Experiment(fid).process()
#
# Use ``quadrature="states-tppi"`` if the acquisition encodes sign alternation
# in F1. Both choices retain the two stored branches at import and use the
# physical time grids. Processing applies the declared NMRPipe input
# convention and performs F2 before F1. Only States-TPPI applies F1
# alternation.
#
# The default ``quadrature="auto"`` uses recognized pulse-program sources.
# Supply the source used for the acquisition with ``pulse_program_source``,
# or put it next to ``fid`` and ``procpar`` with the name given by ``seqfil``
# and a ``.c`` suffix. Check the selected quadrature
# in the reader's log. Choose ``"states"`` or ``"states-tppi"`` explicitly
# when you know the acquisition convention and automatic selection needs it.
#
# Reading processed NMR JCAMP-DX
# ------------------------------
#
# Use the core reader for supported processed 1D spectra and real 2D
# NTUPLES spectra. The 2D path requires nmrglue-ng 1.3 or later.

# %% tags=["skip-execution"]
spectrum = scp.jcamp.read("spectrum.jdx", jcamp_kind="nmr")

# %% [raw] raw_mimetype="text/restructuredtext"
# Omit ``jcamp_kind`` when the file identifies the technique unambiguously.
#
# .. _reading-exchange-formats-through-interop:
#
# Reading exchange formats through interop
# ----------------------------------------
#
# All three examples pass ``protocol="interop"`` to select the
# `nmrglue-ng interop reader <https://nmrglue-ng.readthedocs.io/latest/tutorials/interop.html>`_.
# This route reads and normalizes one file and converts it to an
# ``NDDataset``. With only a path,
# :py:func:`scp.nmr.read() <spectrochempy_nmr.read>` uses its regular reader
# dispatch. The ``.ft2``, ``.jdx`` and ``.nmrML`` files below fall through
# to the TopSpin reader in that dispatch.
#
# **NMRPipe.** Read a processed 2D spectrum from ``spectrum.ft2``.
# Once the interop route is selected, nmrglue-ng can identify the NMRPipe
# profile from the file header, so this call can omit ``profile``.
# The interop reader requires an explicit profile for the JCAMP-DX and
# nmrML examples below.

# %% tags=["skip-execution"]
pipe_spectrum = scp.nmr.read("spectrum.ft2", protocol="interop")

# %% [raw] raw_mimetype="text/restructuredtext"
# **NMR JCAMP-DX.** Pass ``profile="jcampdx-spectrum-1d"`` to select the
# interop reader for a processed 1D NMR spectrum. The ``.jdx`` extension
# identifies the JCAMP-DX container, which can hold data from different
# techniques. For the direct SpectroChemPy route to supported processed
# NMR spectra, use ``scp.jcamp.read()`` as shown above.

# %% tags=["skip-execution"]
jcamp_spectrum = scp.nmr.read(
    "epicatechin.jdx", protocol="interop", profile="jcampdx-spectrum-1d"
)

# %% [raw] raw_mimetype="text/restructuredtext"
# **nmrML.** Pass ``profile="nmrml-1d"`` to select the 1D nmrML reader.
# The ``read_processed=True`` option loads the stored spectrum instead of
# the acquisition data. Use ``axis_endpoint="last-point"`` when the
# producer's ``endValue`` is the coordinate of the last sample. If
# ``endValue`` is the outer edge beyond the last sample, use
# ``"outer-edge"``. This choice sets the spacing of the frequency axis.

# %% tags=["skip-execution"]
xml_spectrum = scp.nmr.read(
    "spectrum.nmrML",
    protocol="interop",
    profile="nmrml-1d",
    read_options={"read_processed": True, "axis_endpoint": "last-point"},
)

# %% [raw] raw_mimetype="text/restructuredtext"
# Use ``read_options`` for file decoding and ``normalize_options`` for
# convention choices, such as the sign of an FID. The optional
# ``conversion_profile`` confirms an interpretation in ``from_interop()``,
# such as ``"states-tppi"`` for adjacent NMRPipe branches with alternation
# still encoded. The reader does not run an FFT. Unsupported profiles or
# missing convention choices raise the corresponding provider or converter
# error. The nmrML reader requires ``xmltodict``.
#
# See :doc:`../nmrpipe_interop` and :doc:`../nmrml_interop` for the underlying
# profiles and their required convention choices.
#
# SIMPSON
# -------
#
# Use ``scp.nmr.read_simpson(...)`` for SIMPSON data files or a simulation
# input file with a sibling output. Supported standalone FIDs use calibrated
# time coordinates. Supported standalone spectra use frequency coordinates
# in Hz. The direct FFT is available for supported 2D FIDs. The indirect quadrature convention
# has yet to be established. When reading a simulation input file, keep
# its output in the same directory.
#
# Additional formats available in nmrglue-ng
# ------------------------------------------
#
# The following readers can be called directly with ``import nmrglue as ng``.
# Their results use the nmrglue-ng API. If you need one of these formats in
# ``scp.nmr.read()``, open a request in the
# `NMR plugin issue tracker <https://github.com/spectrochempy/spectrochempy_nmr/issues>`_
# and describe the acquisition and the workflow you need.
#
# .. list-table::
#    :header-rows: 1
#
#    * - Format
#      - Direct reader
#    * - Tecmag TNMR
#      - ``ng.tecmag.read(path)``
#    * - Magritek Spinsolve
#      - ``ng.spinsolve.read(path)``
#    * - RS2D
#      - ``ng.rs2d.read(path)``
#    * - Sparky UCSF
#      - ``ng.sparky.read(path)``
#    * - Rowland NMR Toolkit
#      - ``ng.rnmrtk.read(path)``
#
# Previously saved SpectroChemPy datasets from Tecmag retain their metadata
# interpretation.
