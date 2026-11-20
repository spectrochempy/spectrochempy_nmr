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
# Indirect quadrature and Fourier processing
# ===========================================
#
# Read the acquisition mode from the normalized dataset, rather than inferring
# it from the experiment name. The reader arranges the stored branches; the
# common processor handles their reconstruction. Begin with the complete
# :doc:`processing_2d` workflow, then use this page to understand F1.
# The following mode-specific recipes use placeholder paths. They explain
# alternative processing sequences, not additional runnable gallery examples.
# Transform F2 before F1 and retain both branches until reconstruction.
# Apply any required digital-filter correction once; see :doc:`replay_2d`.

# %%
import spectrochempy as scp

# %% [raw] raw_mimetype="text/restructuredtext"
# QF magnitude
# ~~~~~~~~~~~~
#
# See :doc:`qf_2d` for a complete, instrument-independent worked workflow.
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
# normalized Rance–Kay arrangement, the plugin first forms ``E - A`` and
# ``1j * (E + A)`` from the two stored branches. For instance,
# ``E = 2 + 3j`` and ``A = 1 + 1j`` give ``1 + 2j`` and ``-4 + 3j``.
# After F2, the two reconstructed branches are cross-paired by real and
# imaginary parts as for States and transformed along F1.

# %% tags=["skip-execution"]
fid = scp.nmr.read_topspin("path/to/echo-antiecho/ser")
spectrum = scp.nmr.Experiment(fid).process(digital_filter="phase")

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
# See the frequency calibration section below for the Agilent TPPI grid.

# %% tags=["skip-execution"]
fid = scp.nmr.read_agilent("path/to/tppi.fid")
spectrum = scp.nmr.Experiment(fid).process()


# %% [raw] raw_mimetype="text/restructuredtext"
# Frequency calibration for Agilent TPPI
# --------------------------------------
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
# %% [raw] raw_mimetype="text/restructuredtext"
# Related concepts
# -----------------
#
# See :doc:`/concepts/quadrature` for acquisition metadata,
# :doc:`/concepts/hypercomplex` for the component representation, and
# :doc:`fourier` for the core Fourier operations.
# The :doc:`synthetic quadrature comparison
# </examples/gallery/nmr/plot_processing_nmr_2d>` illustrates all five modes.
#
# .. toctree::
#    :hidden:
#    :maxdepth: 1
#
#    /concepts/quadrature
#    /concepts/hypercomplex
