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
# The reader arranges the components and records the physical quadrature.
# The common processing path uses this information, independently of the
# instrument, source format, or experiment name (COSY, HSQC, and so on).
# See :doc:`/concepts/quadrature` for the supported conventions and
# :ref:`nmr-processing-validation` for the distinction between runtime checks
# and qualification against reference data.
#
# For an interrupted States-TPPI acquisition or a nonuniformly sampled
# Bruker SER, see :doc:`bruker_indirect_sampling` before transforming F1.
#
# Process the direct dimension (F2) before the indirect dimension (F1) for
# supported paired acquisitions. The plugin uses the acquisition metadata to
# select the quadrature convention and check the transform order. When
# processing explicitly on a supported dataset, you can inspect the
# mixed-domain result after F2 before transforming F1.

# %% [raw] raw_mimetype="text/restructuredtext"
# Process an acquired Bruker HMQC
# --------------------------------
#
# The PST-5 aluminium/phosphorus HMQC in the common datadir has a qualified
# States-TPPI reconstruction. Reading retains its complete acquired pairs,
# corrects the direct digital-filter delay once, and supplies the common
# representation. Apply the windows to the acquired samples before the two
# transforms. This workflow uses SpectroChemPy processing settings, rather
# than replaying the archived TopSpin recipe.

# %%
import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/topspin_2d/1/ser"
raw = scp.nmr.read(path, remove_digital_filter=True)
weighted = raw.sp(ssb=2, pow=2, dim="x").sp(ssb=2, pow=2, dim="y")
spectrum = scp.nmr.Experiment(weighted).process()
magnitude = (spectrum**2).sum(dim="c") ** 0.5
_ = magnitude.plot(
    method="contour",
    start=0.15,
    nlevels=12,
    negative=False,
    xlabel="F2 offset (Hz)",
    ylabel="F1 offset (Hz)",
    figsize=(7, 5),
)

# %% [raw] raw_mimetype="text/restructuredtext"
# The result has four real spectral components. Their Euclidean norm gives
# the displayed hypercomplex magnitude without choosing phase angles.
# Squared cosine windows (``ssb=2, pow=2``) retain the start of each FID.
# Magnitude loses phase and sign information; retain ``spectrum`` and phase
# the components before extracting RR for absorptive peak integration.
# Both coordinates are carrier-relative frequency offsets in Hz. The reader
# retains 73 complete F1 pairs from this interrupted uniform acquisition.
# Windowing and the FFT use that acquired prefix, without constructing
# missing samples. See :doc:`bruker_indirect_sampling` for the stored counts
# and the distinct archived-recipe replay. Data: Christian Fernandez, LCS,
# CC BY 4.0, distributed through spectrochempy_data.
#
# The :doc:`short gallery workflow </examples/gallery/nmr/plot_bruker_2d>`
# runs these same operations. Choose a topic below for the detailed method.


# %% [raw] raw_mimetype="text/restructuredtext"
# Choose a processing topic
# --------------------------
#
# * :doc:`quadrature_2d`: how QF, States, States-TPPI, echo/anti-echo and TPPI
#   encode F1, with Fourier and calibration conventions.
# * :doc:`phase_2d`: automatic common or individual F2 phases and a before/after
#   comparison on a spectral series.
# * :doc:`components_2d`: RR/RI/IR/II planes, slices and replacement of traces.
# * :doc:`baseline`: SNIP in 1D and polynomial corrections along F1 and F2.
# * :doc:`replay_2d`: recorded TopSpin settings and digital-filter policy.
# * :doc:`qf_2d`: a complete QF magnitude workflow on a JEOL COSY acquisition.
#
# .. toctree::
#    :hidden:
#    :maxdepth: 1
#
#    quadrature_2d
#    phase_2d
#    components_2d
#    replay_2d
#    qf_2d
