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
# Check how the indirect dimension was sampled before transforming it.
# An interrupted uniform acquisition contains a contiguous prefix of the
# scheduled increments. A nonuniformly sampled (NUS) acquisition records
# selected increments across the nominal grid. Both can contain fewer
# traces than planned, but their F1 processing differs.
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
# See the `fixture manifest <https://github.com/spectrochempy/spectrochempy_nmr/blob/docs/bruker-f1-workflows/docs/gallery-fixtures.toml>`_
# for the pinned files used by the documentation build.

# %% [raw] raw_mimetype="text/restructuredtext"
# Paired nonuniform sampling
# --------------------------
#
# For supported paired Bruker NUS data, the reader preserves the compact
# acquisition and stores the schedule in ``fid.meta.bruker_nus``.
# ``indices`` contains zero-based F1 indices in storage order.
# ``full_points`` gives the size of the nominal grid in complex F1 points.
# For example, the Bruker NUS acquisition in the
# `NMRPipe demo data <https://www.ibbr.umd.edu/nmrpipe/demo>`_ has 31
# acquired echo/anti-echo pairs on a nominal grid of 62 complex points.
#
# With a valid schedule and indirect spectral width, the F1 coordinate
# contains the acquired times. If the schedule is missing or unusable,
# ``indices`` is ``None``. If the nominal grid cannot be determined,
# ``full_points`` is also ``None``. Inspect the original ``nuslist``
# and acquisition parameters before attempting reconstruction.
#
# F2 can be transformed while the F1 increments remain in acquisition
# order. A regular FFT along F1 is refused because adjacent stored
# increments need not be adjacent in time. A reconstruction method
# must use the schedule and the nominal grid. The direct FFT below
# transforms the stored samples. See :doc:`processing_2d` for the
# Bruker digital-filter options.
#
# .. code-block:: python
#
#    import spectrochempy as scp
#
#    fid = scp.nmr.read_topspin("path/to/nus/ser")
#    print(fid.meta.bruker_nus)
#    f2 = fid.fft(dim="x")
