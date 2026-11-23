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
# Open your NMR data
# ==================
#
# If your spectrum is already processed in the spectrometer software, open
# it and plot it directly. If you have a recorded signal (a FID), read it
# first, then follow :doc:`processing_1d` or :doc:`processing_2d`.
#
# Open a processed Bruker spectrum
# --------------------------------
#
# This example uses the same public data as the
# :doc:`first recipe </getting_started/quickstart>`.

# %%
import spectrochempy as scp

spectrum = scp.nmr.read_topspin(
    "nmrdata/bruker/tests/nmr/topspin_1d", expno=1, procno=1
)
_ = spectrum.real.plot(xlim=(-2.0, 12.0))

# %% [raw] raw_mimetype="text/restructuredtext"
# To open your own experiment, replace the quoted path by the folder
# containing your experiment numbers. ``expno`` chooses the experiment and
# ``procno`` chooses its processed spectrum. For example, use forward slashes
# even on Windows: ``"C:/Users/YourName/NMR/sample"``. Keep the quotes.
#
# Open a recorded signal
# ----------------------
#
# Choose the call for your instrument and replace the path. These are
# alternatives: run only the one matching your file.
#
# **Bruker:** choose the ``fid`` file inside the numbered experiment folder.

# %% tags=["skip-execution"]
fid = scp.nmr.read("C:/Users/YourName/NMR/sample/1/fid")

# %% [raw] raw_mimetype="text/restructuredtext"
# **Agilent/Varian:** choose the acquisition folder containing ``fid``
# and ``procpar``.

# %% tags=["skip-execution"]
fid = scp.nmr.read_agilent("C:/Users/YourName/NMR/sample.fid")

# %% [raw] raw_mimetype="text/restructuredtext"
# **JEOL:** choose the ``.jdf`` acquisition file.

# %% tags=["skip-execution"]
fid = scp.nmr.read_jeol("C:/Users/YourName/NMR/sample.jdf")

# %% [raw] raw_mimetype="text/restructuredtext"
# After reading a 1D FID, follow :doc:`processing_1d` to produce a spectrum.
# For a file that is already a spectrum, plot it directly.
#
# Open a saved SpectroChemPy spectrum
# -----------------------------------
#
# Use the name you chose when saving it, including the ``.scp`` extension.

# %% tags=["skip-execution"]
spectrum = scp.nmr.read("my-spectrum.scp")
_ = spectrum.real.plot()

# %% [raw] raw_mimetype="text/restructuredtext"
# Next steps
# ----------
#
# * :doc:`exchanging_spectra`: NMRPipe, nmrML, JCAMP-DX and CSDF exchanges.
# * :doc:`reader_workflows`: worked Agilent, SIMPSON and nmrML examples.
# * :doc:`/reference/readers`: options for each reader.
#
# If reading fails, check that the path exists and that the acquisition folder
# includes its parameter files. For a 2D acquisition whose mode is unknown,
# do not guess a mode to make processing run; check how it was acquired.
# Format-specific conventions and supported cases are collected in the
# :doc:`technical appendices </appendices/index>`.
