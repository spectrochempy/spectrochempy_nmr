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
# Display a 2D spectrum from a public Bruker acquisition. Run the cells in
# order. This example shows a magnitude map, which is useful for seeing the
# peaks without choosing phase angles. It is not a substitute for a phased
# absorption spectrum when measuring signed peak areas.
#
# The plugin handles the acquisition mode when it is known from the file.
# For an interrupted or nonuniformly sampled acquisition, follow
# :doc:`bruker_indirect_sampling` before processing your own data.
#
# %% [raw] raw_mimetype="text/restructuredtext"
# Process an acquired Bruker HMQC
# -------------------------------
#
# This PST-5 aluminium/phosphorus HMQC is included in the public example
# data. Read the signal, smooth it along both axes, then transform it into
# a spectrum. The map below displays frequency offsets in Hz.

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
# The contours show the positions and relative heights of the visible
# resonances. ``start=0.15`` hides low-level contours below 15% of the
# maximum; lower it to see weaker peaks, or raise it to hide more noise.
# ``nlevels`` sets the number of contour levels. The axes show offsets in Hz.
#
# This map is a magnitude: it does not retain the sign of each resonance.
# Keep ``spectrum`` for phase correction before quantitative integration.
# Data: Christian Fernandez, LCS, CC BY 4.0, distributed through
# spectrochempy_data.
#
# Continue with :doc:`phase_2d` to adjust phase, :doc:`components_2d` to
# inspect sections, or :doc:`analysis` to measure a phased spectrum.
# Acquisition-specific conventions and the reference dataset are described
# in the :doc:`technical appendices </appendices/index>`.
