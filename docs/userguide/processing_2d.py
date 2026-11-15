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
# Processing 2D NMR data
# ======================
#
# Use the plugin's :doc:`reader <reading_data>` to load the 2D dataset.
# The acquisition and quadrature determine which operations are available.
# See :doc:`/concepts/quadrature` for the supported conventions and
# :ref:`nmr-processing-validation` for the distinction between runtime checks
# and qualification against reference data.
#
# Process the direct dimension (F2) before the indirect dimension (F1) for
# supported paired acquisitions. The plugin uses the acquisition metadata to
# select the quadrature convention and check the transform order. When
# processing explicitly on a supported dataset, you can inspect the
# mixed-domain result after F2 before transforming F1.

#
# Transform F2 and F1
# -------------------
#
# Transform the direct time axis ``x`` first when working step by step.
# For paired acquisitions, retain both complex branches through the
# indirect transform. The examples below show how to select a workflow
# for an acquisition whose mode is known. Replace the example paths with
# your own data.

# %%
import spectrochempy as scp

# %% [raw] raw_mimetype="text/restructuredtext"
# QF magnitude
# ~~~~~~~~~~~~
#
# QF records one trace per F1 increment. There is no cosine/sine pair
# to combine. For the supported Bruker COSY replay, transform the complex
# data in F2 and F1, then take the magnitude. A cosine at frequency f
# contributes at both +f and −f. Magnitude display does not recover the
# missing sign of the indirect frequency.

# %% tags=["skip-execution"]
fid = scp.nmr.read_topspin("path/to/qf-cosy/ser")
spectrum = scp.nmr.Experiment(fid).process(replay="topspin")

# %% [raw] raw_mimetype="text/restructuredtext"
# States
# ~~~~~~
#
# States records two traces per increment, a cosine branch C and a sine
# branch S. After F2, combine the real parts as
# ``C.real + 1j * S.real`` and the imaginary parts as
# ``C.imag + 1j * S.imag``. Transform these two complex sequences in F1.
# For example, if the two F2 values at a given point are ``3 + 4j`` and
# ``5 + 6j``, the F1 inputs are ``3 + 5j`` and ``4 + 6j``.
# The plugin handles the branch order and Fourier sign.

# %% tags=["skip-execution"]
fid = scp.nmr.read_topspin("path/to/states/ser", remove_digital_filter=True)
f2 = fid.fft(dim="x")
spectrum = f2.fft(dim="y")

# %% [raw] raw_mimetype="text/restructuredtext"
# States-TPPI
# ~~~~~~~~~~~
#
# The same two branches are acquired, but their signs alternate between
# successive F1 increments. Undo the alternation by multiplying both
# reconstructed sequences by +1, −1, +1, −1 before the F1 FFT.
# At an odd increment, the example States inputs ``3 + 5j`` and
# ``4 + 6j`` would be stored as ``-3 - 5j`` and ``-4 - 6j``.
# The plugin applies the sign correction when the mode is States-TPPI.

# %% tags=["skip-execution"]
fid = scp.nmr.read_topspin("path/to/states-tppi/ser", remove_digital_filter=True)
spectrum = fid.fft(dim="x").fft(dim="y")

# %% [raw] raw_mimetype="text/restructuredtext"
# Echo/anti-echo
# ~~~~~~~~~~~~~~
#
# Echo/anti-echo records two coherence pathways in each increment. For the
# Bruker Rance–Kay profile, the plugin first forms ``E - A`` and
# ``1j * (E + A)`` from the two stored branches. For instance,
# ``E = 2 + 3j`` and ``A = 1 + 1j`` give ``1 + 2j`` and ``-4 + 3j``.
# After F2, the two reconstructed branches are cross-paired by real and
# imaginary parts as for States and transformed along F1.

# %% tags=["skip-execution"]
fid = scp.nmr.read_topspin("path/to/echo-antiecho/ser")
spectrum = scp.nmr.Experiment(fid).process(
    echo_antiecho="bruker-rance-kay", digital_filter="phase"
)

# %% [raw] raw_mimetype="text/restructuredtext"
# TPPI
# ~~~~
#
# Native Agilent TPPI records one trace per increment and advances the
# phase systematically with F1 evolution time. After F2, the real and
# imaginary parts of the traces form separate real-valued F1 sequences.
# Each sequence is zero-filled from N to 2N
# points, and the F1 transform retains bins 0 through N−1. For N=128,
# each real transform therefore uses 256 points and keeps 128 bins.
# See the Agilent TPPI section below for the frequency calibration.

# %% tags=["skip-execution"]
fid = scp.nmr.read_agilent("path/to/tppi.fid")
spectrum = scp.nmr.Experiment(fid).process()

# %% [raw] raw_mimetype="text/restructuredtext"
# The :doc:`gallery example </examples/gallery/nmr/plot_processing_nmr_2d>`
# reconstructs a simulated resonance in all five modes. To inspect a
# complete COSY plane, load the processed Bruker gossypol spectrum from the
# :doc:`NMRPipe example </examples/gallery/importer/plot_read_nmrpipe_2d>`.
# Start the positive contours at 0.3% of the maximum and double each
# successive level to show weak cross-peaks alongside the diagonal.

# %%
path = scp.preferences.datadir / "nmrdata/nmrpipe/nmrpipe_2d_cosy_example.ft2"
cosy = scp.nmr.read(path, protocol="interop")
first_level = 0.003 * cosy.max()
levels = [first_level * 2**index for index in range(8)]
_ = cosy.plot(method="contour", levels=levels, figsize=(6, 6))

# %% [raw] raw_mimetype="text/restructuredtext"
# See :doc:`fourier` for the core FFT and magnitude operations.
#
# A direct FFT uses the supplied samples, including the original group delay
# when reading a Bruker SER without import correction. TopSpin XFB replay applies
# the recorded correction for supported acquisitions. For echo/anti-echo,
# ``Experiment.process(echo_antiecho="bruker-rance-kay", digital_filter="phase")``
# compensates the digital-filter delay in F2. Use ``digital_filter="time"`` for
# time-domain correction instead. The ordinary QF, States, and States-TPPI
# FFT workflows transform the supplied traces without filter correction.
# For these workflows, or for a direct FFT script that used the former
# import correction, pass ``remove_digital_filter=True`` to the reader.
#
# Here ``x`` and ``y`` are the direct and indirect dimensions of the
# imported FID. Keep both quadrature branches through the indirect FFT.
# Select the desired spectral component before plotting or extracting a
# 1D slice.
# Use :doc:`analysis` to find and integrate cross-peaks in a processed spectrum.
#
# For a supported uniform Bruker SER acquisition, use the recorded TopSpin
# processing parameters explicitly:

# %% tags=["skip-execution"]
fid = scp.nmr.read_topspin(
    "path/to/experiment/ser", remove_digital_filter=False, remove_dc_offset=False
)
spectrum = scp.nmr.Experiment(fid).process(replay="topspin")

# %% [raw] raw_mimetype="text/restructuredtext"
# XFB replay supports indirect QF magnitude, States-TPPI, and echo/anti-echo
# with direct DQD. Use the raw SER from the same experiment as the recorded TopSpin processing
# settings. The :ref:`XFB qualification notes <topspin-xfb-replay>` list the
# supported settings. The ordinary ``Experiment.process()`` path
# applies SpectroChemPy's processing sequence to supported acquisitions.
#
# For the underlying dataset, plotting, and general processing API, see
# the `NMR plugin overview in the SpectroChemPy core documentation
# <https://www.spectrochempy.fr/latest/userguide/plugins/nmr.html>`_.
#
# Extracting spectral components
# ------------------------------
#
# Use an ``Experiment`` to extract real 2D planes with the original F1 and F2
# coordinates and intensity units.

# %% tags=["skip-execution"]
experiment = scp.nmr.Experiment(spectrum)
rr = experiment.rr()  # or experiment.RR
ri = experiment.RI
ir = experiment.IR
ii = experiment.II

# %% [raw] raw_mimetype="text/restructuredtext"
# The names follow the TopSpin files ``2rr``, ``2ri``, ``2ir`` and ``2ii``.
# In this convention the first letter refers to F2 and the second to F1.
# A real-only spectrum exposes RR. Requesting a component absent from the
# spectrum raises an error.
#
# .. _topspin-slice-and-baseline:
#
# TopSpin-style slice and baseline commands
# -----------------------------------------
#
# Wrap a single dataset in :class:`~spectrochempy_nmr.Experiment`. These
# commands return new datasets in memory. They use TopSpin numbering, so row
# or column **1** corresponds to index **0** in SpectroChemPy.

# %% tags=["skip-execution"]
fid = scp.nmr.Experiment(fid).rser(1)
experiment = scp.nmr.Experiment(spectrum)
row = experiment.rsr(1)
column = experiment.rsc(1)

# %% [raw] raw_mimetype="text/restructuredtext"
# For a plain F1 × F2 matrix, ``rser(1)`` and ``rsr(1)`` select
# ``dataset[0, :].squeeze("y")``. For a real spectrum, ``rsc(1)`` selects
# ``dataset[:, 0].squeeze("x")``. In a paired acquisition, ``rser()`` numbers the
# two stored traces of each F1 increment consecutively. With four spectral
# components, ``rsr()`` keeps the complex F2 trace from the F1-real channel, and
# ``rsc()`` builds a complex F1 trace from the F2-real components. After an
# F2-only transform, ``rsc()`` returns an F1 time trace with the acquisition
# convention still encoded.
#
# Use ``wsr()`` or ``wsc()`` to replace a trace in a copy of the spectrum.
# The replacement must have the same coordinate grid and units.

# %% tags=["skip-execution"]
row = experiment.rsr(1)
adjusted = row.pk(phc0=10.0)
updated = experiment.wsr(1, adjusted)

# %% [raw] raw_mimetype="text/restructuredtext"
# On a real matrix the corresponding assignment is ``updated[0, :] = adjusted``
# or ``updated[:, 0] = adjusted``, using a copy of the original spectrum.
# On a Bruker four-component spectrum, ``wsr()`` replaces the selected row in
# ``2rr`` and ``2ir``, and ``wsc()`` replaces the selected column in ``2rr``
# and ``2ri``.
#
#
# Use ``abs2()`` to fit a polynomial separately along each F2 row and
# ``abs1()`` to fit along each F1 column of a fully transformed spectrum.
# Both correct only the real-real plane, as in TopSpin. The default polynomial
# degree is 5. Set ``limits`` to correct a specific interval and ``ranges`` to
# identify peak-free regions for the fit. Both take values in the units of
# the selected axis. By default the fit iteratively excludes positive and negative peaks
# using each trace's residual noise estimate.

# %% tags=["skip-execution"]
corrected = experiment.abs2(
    order=1, limits=(20.0, 65.0), ranges=[(21.0, 24.0), (61.0, 64.0)]
)
corrected = scp.nmr.Experiment(corrected).abs1(order=1)

# %% [raw] raw_mimetype="text/restructuredtext"
# The fit excludes masked points, including when fit ranges are supplied.
# A correction interval requires a strictly monotonic axis. Automatic fit-point
# selection can differ from TopSpin's peak-region selection.
#
# Agilent TPPI
# ------------
#
# For a native Agilent acquisition identified as TPPI, use
# ``Experiment(fid).process()``. The result contains four planes in F1, F2
# component order (``rr``, ``ri``, ``ir``, ``ii``). The direct transform
# uses the sample convention of ``var2pipe -noaswap`` followed by
# ``FT -nodf``. In F1 the real and imaginary parts of F2 each form a
# real sequence. Each sequence is zero-filled from N to 2N, transformed,
# and reduced to bins 0 through N−1. This retains DC and excludes Nyquist.
# The F1 bin spacing is the acquisition width divided by 2N. The displayed
# half-band follows the carrier convention of NMRPipe
# ``TP → ZF -size 2N → FT -real``, with the carrier halfway between its
# two central bins.
#
# The default axes are carrier-relative Hz. To obtain ppm, pass the
# observation frequency in MHz and carrier in ppm explicitly for each
# dimension, in F1 then F2 order. For the historical Agilent
# ``agilent_2d_tppi`` NMRPipe conversion the values are:

# %% tags=["skip-execution"]
spectrum = scp.nmr.Experiment(fid).process(
    tppi_reference=((125.681, 101.274), (125.681, 101.274))
)

# %% [raw] raw_mimetype="text/restructuredtext"
# Use the values from the reference for your acquisition. The Agilent
# reader can lack the F1 observation frequency.
#
# Automatic phase correction along F2
# -----------------------------------
#
# For a series of complex F2 spectra, ``apk2d()`` estimates a common phase
# from the first row by default. Pass ``reference_row`` to select another row
# using TopSpin's 1-based numbering. Use ``per_row=True`` when the phase varies
# across the series. Inspect individual estimates when the signal becomes
# weak, especially at high diffusion weightings. The estimator uses the same options as ``scp.nmr.apk()``.

# %% tags=["skip-execution"]
phased = scp.nmr.Experiment(series).apk2d(reference_row=1, order=0)  # noqa: F821

# %% [raw] raw_mimetype="text/restructuredtext"
# For MQMAS, pass the complex result of ``xf2()``. The common phase applies
# to both States branches. The output can go directly to ``xfshear()``.
# An unsuccessful estimate raises an error identifying the row.
#
# Related concepts
# ----------------
#
# .. toctree::
#    :maxdepth: 1
#
#    /concepts/quadrature
#    /concepts/hypercomplex
