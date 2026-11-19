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
# Recorded TopSpin processing and digital filters
# ===============================================
#
# The common processor can explicitly replay a compatible recorded TopSpin
# recipe. This is a separate choice from the workflow in :doc:`processing_2d`,
# where windowing and transforms are selected explicitly.
#
# Correct the digital filter once
# --------------------------------
# A direct FFT uses the supplied samples, including the original group delay
# when reading a Bruker SER without import correction. TopSpin XFB replay applies
# the recorded correction for supported acquisitions. For echo/anti-echo,
# ``Experiment.process(digital_filter="phase")``
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

# %%
import spectrochempy as scp

fid = scp.nmr.read_topspin(
    scp.preferences.datadir / "nmrdata/bruker/tests/nmr/topspin_2d/1/ser",
    remove_digital_filter=False,
    remove_dc_offset=False,
)
spectrum = scp.nmr.Experiment(fid).process(replay="topspin")
rr = scp.nmr.Experiment(spectrum).RR
_ = rr.plot(method="contour", start=0.06, nlevels=12, negative=False)

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
