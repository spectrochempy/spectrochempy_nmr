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
# Bruker SER acquisitions with incomplete F1 sampling
# ===================================================
#
# Reconstruct a complete 2D spectrum from an interrupted uniform
# acquisition. This workflow uses the acquired prefix and the archived
# processing parameters to replay TopSpin XFB and plot the real-real plane.
#
# Interrupted uniform States-TPPI
# -------------------------------
#
# In a paired States-TPPI acquisition, each complex F1 point uses two
# direct traces. For supported uniform Bruker States-TPPI SER files, the
# reader uses the recorded indirect acquisition count (``acqu2s.TD``)
# to identify the acquired prefix when the remaining stored traces are
# zero-filled. The reader omits a final unpaired trace and exposes the
# number of complete complex points as ``fid.meta.tdeff[-2]``. It retains
# the archived acquisition and processing parameters.
#
# The ``interrupted_states_tppi`` metadata gives counts in **traces**:
# ``stored_traces`` includes the zero-filled tail, ``acquired_traces``
# includes any final unpaired trace, and ``paired_traces`` counts the
# traces retained in complete pairs. For example, 147 acquired traces
# in a SER with 192 stored traces give 146 paired traces and 73 complex
# F1 points.
#
# Read the original SER before explicitly replaying TopSpin XFB. Replay
# requires compatible archived processing settings, including the
# effective indirect trace count. Leave digital-filter and DC-offset
# corrections to the replay. See :ref:`topspin-xfb-replay` for the supported
# settings and :doc:`processing_2d` for explicit Fourier processing.

# %%
import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/topspin_2d/1/ser"
fid = scp.nmr.read_topspin(path)
print(fid.meta.interrupted_states_tppi, fid.meta.tdeff[-2])
spectrum = scp.nmr.Experiment(fid).process(replay="topspin")
rr = scp.nmr.Experiment(spectrum).rr()
_ = rr.plot(
    method="contour",
    start=0.06,
    nlevels=12,
    negative=False,
    cmap="viridis",
    figsize=(7, 5),
)

# %% [raw] raw_mimetype="text/restructuredtext"
# The plot starts at 6% of the maximum and uses 12 automatically generated
# positive contour levels.
#
# The contours show the real-real plane of the interrupted PST-5 HMQC
# acquisition after XFB replay. The acquisition correlates aluminium-27
# and phosphorus-31. The reader retains 146 traces, giving 73 complete
# complex F1 points. The displayed grid follows the archived TopSpin
# processing sizes and references. Zero filling increases the displayed
# grid size while the number of acquired points remains 73.
#
# The measurement and parameters are attributed to Christian Fernandez,
# LCS, and distributed under CC BY 4.0 through
# `spectrochempy_data <https://github.com/spectrochempy/spectrochempy_data/tree/08bb9b0cbff8f4363c48b6bff7ccb743f8140e0a/testdata/nmrdata/bruker/tests/nmr/topspin_2d/1>`_.
# See the `fixture manifest <https://github.com/spectrochempy/spectrochempy_nmr/blob/master/docs/gallery-fixtures.toml>`_
# for the pinned files used by the documentation build.
