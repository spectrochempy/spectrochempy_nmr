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
# Complete reader workflows
# =========================
#
# Start with ``scp.nmr.read()`` and pass the resulting dataset to
# ``scp.nmr.Experiment``. Reading decodes the file, arranges the components
# and supplies their coordinates and conventions. Processing then consumes
# that representation. A 1D FID needs a Fourier transform; an already processed
# spectrum can be plotted directly. See :doc:`reading_data` for reader options
# and :doc:`processing_2d` for acquired Bruker 2D data.
#
# Prepare the synthetic inputs
# -----------------------------
#
# The three short reader examples use deterministic, synthetic 1D FIDs.
# They demonstrate complete file-reading and processing workflows without
# requiring a spectrometer or an external simulator. They do not establish
# experimental accuracy or support for other acquisition layouts.
#
# Download :download:`generate_reader_inputs.py <../generate_reader_inputs.py>`
# and run it once in the same Python environment as SpectroChemPy::
#
#     python generate_reader_inputs.py
#
# The script writes below ``scp.preferences.datadir / "nmrdata/synthetic"``.
# It replaces only its own four files: Agilent ``fid`` and ``procpar``,
# SIMPSON ``simpson.fid`` and nmrML ``signal.nmrML``. The documentation's
# ``python docs/fetch_gallery_data.py`` command also generates these inputs
# after fetching the measured datasets. No persistent datadir setting changes.
#
# The stored numerical signal has 1024 complex samples, a 4000 Hz sampling
# width, 40 ms exponential decay and a positive 500 Hz phase evolution.
# The script uses nmrglue-ng's writer for the Agilent container, a SIMPSON
# text header and rows, and an nmrML XML document with a base64-encoded
# little-endian complex128 array. NumPy here constructs interchange samples;
# all processing in the examples uses the public SpectroChemPy API.
#
# Agilent/Varian: acquisition to spectrum
# ---------------------------------------
#
# Read the directory containing both ``fid`` and ``procpar``. The reader
# obtains the direct sampling width, nucleus and observation frequency from
# ``procpar``. It recognizes the container without an explicit protocol.
#

# %%
import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/synthetic/agilent.fid"
raw = scp.nmr.read(path)
spectrum = scp.nmr.Experiment(raw).process()
_ = spectrum.real.plot(xlim=(2.0, 0.5), title="Synthetic Agilent/Varian spectrum")

# %% [raw] raw_mimetype="text/restructuredtext"
# The peak appears near 1.25 ppm on this synthetic 400 MHz acquisition's
# native display grid. Its carrier is zero. This is a 1D example; for native
# 2D States, States-TPPI and TPPI acquisitions, use the mode and calibration
# requirements in :doc:`reading_data` and :doc:`processing_2d`.
#
# See the :doc:`runnable gallery example
# </examples/gallery/importer/plot_read_agilent>` for its output figure.
#
# SIMPSON: text FID to spectrum
# ------------------------------
#
# A standalone SIMPSON text FID declares its point count, sampling width
# and ``TYPE=FID``. Each row contains the real and imaginary parts of one
# sample. Reading does not execute a SIMPSON input program.
#

# %%
path = scp.preferences.datadir / "nmrdata/synthetic/simpson.fid"
raw = scp.nmr.read(path)
spectrum = scp.nmr.Experiment(raw).process()
_ = spectrum.real.plot(xlim=(-800, -200), xlabel="Frequency offset (Hz)")

# %% [raw] raw_mimetype="text/restructuredtext"
# With this file's stored phase evolution and the native SIMPSON Fourier
# convention, the peak is at -500 Hz. The output is in frequency offsets;
# the standalone file supplies no chemical-shift reference. A file already
# declared ``TYPE=SPE`` contains a spectrum and must not be transformed again.
# For the output figure, see the :doc:`gallery example
# </examples/gallery/importer/plot_read_simpson>`.
#
# nmrML: declare the raw interpretation
# -------------------------------------
#
# Install the provider's optional XML decoder if it is absent::
#
#     python -m pip install xmltodict
#
# The XML declaration counts real and imaginary scalars, so 2048 declared
# values represent 1024 complex samples. Pass ``raw_point_count="scalar"``
# explicitly. The document also declares uniform sampling and a width in Hz.
# ``normalize_options`` chooses the stored signal convention, independently
# of the XML container. Here ``conjugation="none"`` retains the synthetic
# positive phase evolution.
#

# %%
path = scp.preferences.datadir / "nmrdata/synthetic/signal.nmrML"
raw = scp.nmr.read(
    path,
    protocol="interop",
    profile="nmrml-1d",
    read_options={"raw_point_count": "scalar"},
    normalize_options={"conjugation": "none"},
)
spectrum = scp.nmr.Experiment(raw).process()
_ = spectrum.real.plot(xlim=(200, 800), xlabel="Frequency offset (Hz)")

# %% [raw] raw_mimetype="text/restructuredtext"
# The peak is at +500 Hz. Keep the coordinate in Hz unless the observation
# frequency and reference are known. To read an existing stored nmrML spectrum,
# select ``read_processed=True`` and supply its axis-endpoint convention as
# explained in :doc:`reading_data`; that is a different selection from this
# FID example. See :doc:`../nmrml_interop` for the supported profiles and the
# :doc:`gallery example </examples/gallery/importer/plot_read_nmrml>` for
# the executed output.
#
