# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Read a Bruker TopSpin 1D FID
============================

In this example, we load a 1D Bruker TopSpin NMR dataset and inspect the raw
FID with the public ``scp.nmr.read(...)`` API.

"""

# %%
import spectrochempy as scp

path = "nmrdata/bruker/tests/nmr/topspin_1d"

# %%
# load the data in a new dataset

ndd = scp.nmr.read(path, expno=1, remove_digital_filter=True)

# %%
# view it...

_ = ndd.plot()

# %%
# Uncomment the following line to display all figures when running the script
# directly with Python.

# %%

# scp.show()
