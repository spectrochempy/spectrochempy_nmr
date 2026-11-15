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
# .. _validated-1d-processing:
#
# Processing 1D NMR data
# ======================
#
# .. _nmr-processing-validation:
#
# Processing, validation, and qualification
# -----------------------------------------
#
# * **Processing** means applying operations to the data. For example, apply
#   exponential apodization and then an FFT to a time-domain ``NDDataset``.
#   When working step by step, you choose the operations and their order.
# * **Validation** is the set of checks made when you call
#   ``Experiment.process()``. The plugin checks whether it can run the
#   requested workflow on that dataset, using properties such as its domain,
#   dimensions and acquisition encoding. Individual core operations also
#   have their own input checks.
# * **Qualification** is evidence from testing a specific acquisition and
#   processing configuration. For example, the TopSpin EFP replay below was
#   compared with a TopSpin reference for the documented Bruker DQD profile.
#   That comparison applies to the settings covered by the reference tests.
#   A successful call to ``Experiment.process()`` tells you the input passed
#   the runtime checks.
#
# Processing a FID
# ----------------
#
# Read the Bruker FID with digital-filter correction disabled to inspect
# the recorded time-domain signal. The plot shows its real component.

# %%
import spectrochempy as scp
from spectrochempy_nmr import autophase

path = "nmrdata/bruker/tests/nmr/topspin_1d/1/fid"
raw_fid = scp.nmr.read(path, remove_digital_filter=False)
_ = raw_fid.plot(title="Bruker FID as recorded")

# %% [raw] raw_mimetype="text/restructuredtext"
# The digital filter delays the signal by 76 samples (about 304 µs for
# this acquisition). Plot the first 480 points to see the delayed onset
# and subsequent decay. With ``remove_digital_filter=True``, the reader
# compensates for the delay and the signal appears near the start of the
# corrected time axis. The magnitude shows the onset independently of
# the phase of the complex FID.

# %%
fid = scp.nmr.read(path, remove_digital_filter=True)

_ = scp.plot_multiple(
    [abs(raw_fid[:480]), abs(fid[:480])],
    method="pen",
    labels=["As recorded", "Delay corrected"],
    legend="upper right",
    xlabel="Time / µs",
    ylabel="FID magnitude / count",
)

# %% [raw] raw_mimetype="text/restructuredtext"
# Wrap the corrected FID in :class:`~spectrochempy_nmr.Experiment` and
# select explicit processing parameters.

# %%
experiment = scp.nmr.Experiment(fid)
result = experiment.process(apodization="em", lb=5.0, phase=None)

# %% [raw] raw_mimetype="text/restructuredtext"
# See :doc:`apodization` for the ``em``, ``gm`` and ``sp`` windows. Use
# ``result.history`` to inspect the steps recorded on the spectrum.
# Use ``phase="manual"`` with explicit angles. With core 1.2.1 and
# ``phase=None``, the transform leaves the phase untouched. The development
# version of the plugin also supports ``phase="metadata"`` to apply angles
# attached to the FID.
#
# To inspect intermediate results, apply the core operations to the
# ``NDDataset`` exposed by ``experiment.dataset``. You can call them as
# dataset methods or as functions such as ``scp.em(experiment.dataset, lb=5.0)``.
# Calling ``plot()`` on the complex spectrum displays its real component by
# default. Set ``show_complex=True`` to add the imaginary component as a
# dashed line.

# %%
windowed = experiment.dataset.em(lb=5.0)
padded = windowed.zf_size(size=max(32768, windowed.shape[-1]))
spectrum = padded.fft()
unphased_axes = spectrum.plot(show_complex=True, title="After Fourier transformation")
unphased_axes.autoscale(axis="y")

# %% [raw] raw_mimetype="text/restructuredtext"
# Apply phase correction to the complex spectrum, then plot both
# components over the same ppm range. The zero-order angle ``phc0`` is constant
# across the spectrum, while the first-order angle ``phc1`` varies linearly
# with frequency. The angles below are in degrees and apply to this
# acquisition. Compare the shape of the main resonance before and after
# correction.

# %%
phased = spectrum.pk(phc0=52.44, phc1=-16.84)
phased_axes = phased.plot(show_complex=True, title="After phase correction")
phased_axes.autoscale(axis="y")

# %% [raw] raw_mimetype="text/restructuredtext"
# Both approaches preserve the original FID by default. A supported complex
# JEOL 1D FID read with ``scp.nmr.read_jeol()`` can use either approach.
# For example, on the epicatechin JDF reference, ``Experiment.process()``
# produces a spectrum referenced in ppm, and the stepwise core workflow above
# produces a frequency axis in Hz.
#
# For a 1D acquisition outside the scope of ``Experiment.process()``, you
# can still apply individual operations when the data meet their input
# requirements. For example, use a suitable time-domain axis for apodization
# and FFT.
#
# An exponential window trades some resolution for sensitivity. Zero filling
# sets the spacing of the frequency bins. See :doc:`analysis` for baseline
# correction, peak picking, integration and fitting of a phased spectrum.
#
# A direct ``.fft()`` transforms the samples supplied to it. For a Bruker FID
# you can explicitly correct the filter at import with
# ``remove_digital_filter=True``. With the development plugin, use
# ``experiment.process(digital_filter="time")`` to correct a supported FID
# as part of processing. For a normalized FID that has not been scaled by
# scan count and receiver gain, ``prepare_fid()`` can also apply DC-offset
# and digital-filter corrections before the FFT.
#
# A pseudo-2D series contains ordinary 1D spectra that can be processed one
# spectrum at a time. For acquisitions with two Fourier dimensions, see
# :doc:`processing_2d`.
#
# See :ref:`nmr-core-operations` for the core functions used in NMR.
#
# Estimate phase and correct the baseline
# ---------------------------------------
#
# The development version of the plugin provides ``autophase()`` for a
# complex spectrum after the FFT. It requires a version of nmrglue-ng with
# phase estimation support.

# %%
diagnostic = autophase(spectrum, order=0, apply=False)
result = autophase(spectrum, order=0, baseline_options={"model": "asls", "lamb": 1e9})

# %% [raw] raw_mimetype="text/restructuredtext"
# Inspect ``result.estimate`` for the correction angles and fit diagnostics.
# ``result.phased`` is the complex spectrum. If you request baseline correction,
# ``result.baseline`` and ``result.corrected`` are real datasets. Use
# ``result.phased`` for further phase adjustments. AsLS is suited to positive
# peaks. For signed peaks, fit a polynomial on known signal-free regions.
#
# ACME assumes positive absorption peaks. Pass a boolean ``exclude`` array to
# omit artifacts from phase scoring while retaining their data in the result.
# For isolated peaks of known signs, choose ``method="dispa"`` and provide
# ``references=[(start, stop, sign), ...]`` with half-open index windows. Use
# well-separated references for order one. The optional
# ``baseline_mask`` selects signal-free indices to fit a provisional complex
# linear background for phase scoring. The final real baseline is a separate
# fit. A successful solver status still calls for inspecting the result.
#
# Replay TopSpin EFP on a Bruker DQD FID
# --------------------------------------
#
# Use the raw FID and the processing status file (``procs``) from a completed
# TopSpin ``efp`` run in the same experiment:

# %% tags=["skip-execution"]
fid = scp.nmr.read_topspin(
    "experiment/1/fid", remove_digital_filter=False, remove_dc_offset=False
)
spectrum = scp.nmr.Experiment(fid).process(replay="topspin")

# %% [raw] raw_mimetype="text/restructuredtext"
# Replay uses the recorded exponential window, FFT size and phase. It applies
# digital-filter compensation on the final FFT grid, including fractional group
# delay, and uses the processing ppm axis. Intensities retain the FID reader's
# scan and receiver gain normalization, so comparison to TopSpin may require an
# overall amplitude factor.
#
# The supported profile is a full-spectrum DQD 1D ``efp`` run with
# ``PKNL`` enabled, ``FCOR=1``, and baseline correction, linear prediction and
# axis reversal disabled.
#
# Fractional PKNL correction for a Bruker QSIM FID
# ------------------------------------------------
#
# Read the original FID with both import corrections disabled. The plugin applies the
# digital-filter correction after the FFT. If you already corrected the
# filter during import, reread the original FID before using PKNL.

# %% tags=["skip-execution"]
fid = scp.nmr.read_topspin(
    "experiment/80/fid", remove_digital_filter=False, remove_dc_offset=False
)
spectrum = scp.nmr.Experiment(fid).process(
    apodization="em", lb=50.0, size=8192, digital_filter="pknl"
)

# %% [raw] raw_mimetype="text/restructuredtext"
# The result uses the processing ppm reference. To apply measured phase
# angles at the first frequency bin, pass ``phase="manual"`` with ``phc0``
# and ``phc1`` in degrees. The
# :doc:`cross-polarization example </examples/gallery/analysis/plot_processing_cp_nmr>`
# uses one pair of angles from a representative FID for the entire series.
