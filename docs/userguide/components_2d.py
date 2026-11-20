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
# Spectral components and slices
# ===============================
#
# A processed 2D dataset can contain several real component planes. Select
# one plane before plotting or integrating. For a complete four-component
# acquisition, start from :doc:`processing_2d` and retain its ``spectrum``.
#
# Display a processed real plane
# ------------------------------
#
# This processed NMRPipe COSY export contains one real plane. It is already
# in the frequency domain; read and plot it directly. The contour levels start
# at 0.3% of the maximum and double successively.

# %%
import spectrochempy as scp

# %%
path = scp.preferences.datadir / "nmrdata/nmrpipe/nmrpipe_2d_cosy_example.ft2"
cosy = scp.nmr.read(path, protocol="interop")
first_level = 0.003 * cosy.max()
levels = [first_level * 2**index for index in range(8)]
_ = cosy.plot(method="contour", levels=levels, figsize=(6, 6))


# %% [raw] raw_mimetype="text/restructuredtext"
# Extracting spectral components
# ------------------------------
#
# Use an ``Experiment`` to extract real 2D planes with the original F1 and F2
# coordinates and intensity units. Read and transform the measured HMQC
# from :doc:`processing_2d` to obtain its four components.

# %%
path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/topspin_2d/1/ser"
fid = scp.nmr.read(path, remove_digital_filter=True)
spectrum = scp.nmr.Experiment(fid).process()
experiment = scp.nmr.Experiment(spectrum)
rr = experiment.rr()  # or experiment.RR
ri = experiment.RI
ir = experiment.IR
ii = experiment.II

# %% [raw] raw_mimetype="text/restructuredtext"
# The names follow the TopSpin files ``2rr``, ``2ri``, ``2ir`` and ``2ii``.
# In this convention the first letter refers to F2 and the second to F1.
# A real-only spectrum exposes RR. Requesting a component absent from the
# spectrum raises an error.
#
# .. _nmr-2d-slices:
#
# Extract and replace slices
# --------------------------
#
# Wrap a single dataset in :class:`~spectrochempy_nmr.Experiment`. These
# commands return new datasets in memory. They use TopSpin numbering, so row
# or column **1** corresponds to index **0** in SpectroChemPy.

# %%
direct_fid = scp.nmr.Experiment(fid).rser(1)
experiment = scp.nmr.Experiment(spectrum)
row = experiment.rsr(1)
column = experiment.rsc(1)

# %% [raw] raw_mimetype="text/restructuredtext"
# For a plain F1 × F2 matrix, ``rser(1)`` and ``rsr(1)`` select
# ``dataset[0, :].squeeze("y")``. For a real spectrum, ``rsc(1)`` selects
# ``dataset[:, 0].squeeze("x")``. In a paired acquisition, ``rser()`` numbers the
# two stored traces of each F1 increment consecutively. With four spectral
# components, ``rsr()`` keeps the complex F2 trace from the F1-real channel, and
# ``rsc()`` builds a complex F1 trace from the F2-real components. After an
# F2-only transform, ``rsc()`` returns an F1 time trace with the acquisition
# convention still encoded.
#
# Use ``wsr()`` or ``wsc()`` to replace a trace in a copy of the spectrum.
# The replacement must have the same coordinate grid and units.

# %%
row = experiment.rsr(1)
adjusted = 0.9 * row
updated = experiment.wsr(1, adjusted)

# %% [raw] raw_mimetype="text/restructuredtext"
# On a real matrix the corresponding assignment is ``updated[0, :] = adjusted``
# or ``updated[:, 0] = adjusted``, using a copy of the original spectrum.
# On a Bruker four-component spectrum, ``wsr()`` replaces the selected row in
# ``2rr`` and ``2ir``, and ``wsc()`` replaces the selected column in ``2rr``
# and ``2ri``.
#
#

# %% [raw] raw_mimetype="text/restructuredtext"
# Baseline and phase
# ------------------
#
# For polynomial corrections along F2 or F1, continue with :doc:`baseline`.
# For automatic F2 phase estimation before selecting a real plane, see
# :doc:`phase_2d`.
