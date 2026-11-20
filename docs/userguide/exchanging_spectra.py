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
# Exchange processed NMR spectra
# ==============================
#
# Use the NMR writers to share spectra after Fourier processing and phase
# correction. Choose a format based on the spectral components, coordinates
# and units the receiving application needs. This guide exports small
# synthetic spectra, reads the files back and compares their contents.
#
# Choose the exchange format
# --------------------------
#
# .. list-table::
#    :header-rows: 1
#    :widths: 25 35 40
#
#    * - Format
#      - Spectrum supported by the writer
#      - Coordinates and intensity units
#    * - JCAMP-DX
#      - One real 1D spectrum, at least two points
#      - Regular axis in Hz or ppm; unassigned intensity units
#    * - CSDF
#      - One real or complex 1D or 2D spectrum
#      - Linear axes in Hz or SI-prefixed Hz; portable SI intensity units
#
# Both writers require unmasked data. For hypercomplex input, select a
# component the receiving format can represent. For example,
# ``Experiment(spectrum).rr()`` extracts a real-real 2D plane.
# CSDF can retain one complex signal. For exchanges with Dmfit on Windows,
# choose CSDF and see the dedicated Dmfit workflow below.
#
# Exchange files carry the samples and selected metadata. To preserve
# a dataset's processing history and vendor metadata for later work in
# SpectroChemPy, save its native SCP file.
#
# Build a real spectrum with a known frequency grid
# -------------------------------------------------
#
# This synthetic line is centered at +250 Hz with a full width at half
# maximum of 40 Hz. Its coordinate decreases from +1000 to -1000 Hz.
# The line shape helper returns a dataset without assigned intensity units,
# suitable for both formats in this example.

# %%
from pathlib import Path
from tempfile import TemporaryDirectory

import spectrochempy as scp

frequency = scp.Coord.linspace(
    1000,
    -1000,
    501,
    units="Hz",
    title="Frequency offset",
    rounding=False,
)
spectrum = scp.lorentzian(frequency, pos=250, width=40, normalized=False)
spectrum.name = "Synthetic 1H spectrum"
_ = spectrum.plot()

# %% [raw] raw_mimetype="text/restructuredtext"
# Export and read JCAMP-DX
# ------------------------
#
# Supply the observation frequency in MHz or as a compatible quantity,
# and a nucleus such as ``"^1H"``. The writer can recover these values from
# available NMR metadata. Supply them explicitly for a synthetic spectrum.
#
# The writer preserves stored Hz or ppm coordinates and their direction.
# Supplying an observation frequency does not recalibrate a Hz axis to a
# chemical-shift reference. Use an explicitly referenced ppm coordinate
# when chemical shifts are needed.
#
# A temporary directory lets repeated runs create fresh files. Replace
# its path with your chosen directory to keep the export. Existing files
# require ``overwrite=True``. The dataset name used as the JCAMP-DX title
# must be printable ASCII.

# %%
with TemporaryDirectory() as directory:
    path = scp.nmr.write_jcamp(
        spectrum,
        Path(directory) / "spectrum.jdx",
        observation_frequency=400,
        nucleus="^1H",
    )
    jcamp = scp.jcamp.read(path, jcamp_kind="nmr")

_ = scp.plot_multiple(
    [spectrum, jcamp], labels=["Original", "JCAMP-DX"], method="pen", legend="best"
)
print("Largest absolute intensity difference:", abs(spectrum - jcamp).max())

# %% [raw] raw_mimetype="text/restructuredtext"
# The curves overlap. The reader returns the stored frequency coordinates
# and observation metadata, with the signal in the frequency domain.
# This round trip checks the local reader/writer pair. For comparisons
# with TopSpin, including integer scaling and numerical formatting,
# see :doc:`../processed_1d_interop`.
#
# Export a complex spectrum with CSDF
# -----------------------------------
#
# The plugin reads and writes a self-contained CSDF 1.0 file containing
# one spectrum. CSDF stores data using the
# `Core Scientific Dataset Model (CSDM)
# <https://csdmpy.readthedocs.io/en/stable/CSD_model.html>`_.
# The `csdmpy documentation <https://csdmpy.readthedocs.io/en/stable/>`_
# describes the model and its Python tools for reading and writing files.
# The plugin supports the subset described here without requiring csdmpy.
# A constant phase below gives the line an imaginary component.
# The round trip retains both components and a portable SI signal unit.
# Accepted storage types are float32, float64, complex64 and complex128.
# Every axis must be a linear frequency grid in Hz or SI-prefixed Hz.

# %%
complex_spectrum = spectrum * (1 + 0.25j)
complex_spectrum.units = "V"
with TemporaryDirectory() as directory:
    path = scp.nmr.write_csdf(complex_spectrum, Path(directory) / "spectrum.csdf")
    csdf = scp.nmr.read_csdf(path)

_ = scp.plot_multiple(
    [csdf.real, csdf.imag], labels=["Real", "Imaginary"], method="pen", legend="best"
)
print("Largest absolute complex difference:", abs(complex_spectrum - csdf).max())

# %% [raw] raw_mimetype="text/restructuredtext"
# ``scp.nmr.read(path)`` also recognizes the .csdf extension.
# The writer includes axis labels, portable SI intensity units and stored
# CSDM origin offsets when available. The reader expresses axes in Hz.
# A stored origin offset alone does not establish a chemical-shift reference.
# Convert a ppm axis to Hz with a known reference before exporting.
#
# Exchange a 2D plane
# -------------------
#
# The same CSDF calls work for a real or complex 2D spectrum. Here two
# independent line shapes form a synthetic real peak. F2 decreases,
# while F1 increases. The outer product constructs the plane on these
# two independent frequency grids.

# %%
f1 = scp.Coord.linspace(-500, 500, 101, units="Hz", title="F1", rounding=False)
f2 = scp.Coord.linspace(1000, -1000, 201, units="Hz", title="F2", rounding=False)
indirect = scp.lorentzian(f1, pos=-120, width=50, normalized=False)
direct = scp.lorentzian(f2, pos=250, width=60, normalized=False)
plane = scp.NDDataset(
    indirect.data[:, None] * direct.data[None, :],
    coordset=[f1, f2],
    units="V",
)
with TemporaryDirectory() as directory:
    path = scp.nmr.write_csdf(plane, Path(directory) / "plane.csdf")
    restored_plane = scp.nmr.read(path)

_ = restored_plane.plot(
    method="contour",
    start=0.1,
    nlevels=12,
    negative=False,
    cmap="viridis",
)
print("Largest absolute plane difference:", abs(plane - restored_plane).max())

# %% [raw] raw_mimetype="text/restructuredtext"
# Exchange spectra with Dmfit on Windows
# --------------------------------------
#
# `Dmfit <https://nmr.cemhti.cnrs-orleans.fr/Dmfit/Default.aspx>`_ is a
# solid-state NMR fitting application for Windows. It documents CSDF input
# and output on its
# `CSDF format page
# <https://nmr.cemhti.cnrs-orleans.fr/Dmfit/Help/FileFormat/CsdfFileFormat.aspx>`_.
#
# To take a processed spectrum from SpectroChemPy to Dmfit, export it with
# ``scp.nmr.write_csdf(spectrum, "spectrum.csdf")``, then open that CSDF
# file in Dmfit. To bring a spectrum back, save it as CSDF in Dmfit and
# load it with ``scp.nmr.read_csdf("spectrum.csdf")``. The supported
# coordinates, units and components are described above.
#
# Two real spectra originating from Dmfit, phosphorus-31 phosphate and
# aluminium-27 YAG, were imported and re-exported with matching samples,
# frequency coordinates and CSDM origins, checked independently with
# `csdmpy <https://csdmpy.readthedocs.io/en/stable/>`_.
# Opening a plugin export in Dmfit itself remains to be qualified.
# See :doc:`../processed_1d_interop` for the files and qualification details.
#
# Check the result in the receiving application
# ---------------------------------------------
#
# Compare the number of points, both coordinate arrays, axis direction,
# units and available components. Plot the reloaded spectrum and inspect
# the difference to detect an axis reversal or a scaling change.
# For complex spectra compare the full complex signal before selecting
# a component for display.
#
# CSDF exchange omits processing history, vendor metadata and reciprocal
# axis metadata. Applications can impose additional requirements. The
# qualified SIMPSON CSDF path, for example, requires increasing axes.
# See :doc:`../processed_1d_interop` for the supported subset and
# application qualifications. The :doc:`reading_data` guide describes
# NMRPipe and nmrML import profiles.
