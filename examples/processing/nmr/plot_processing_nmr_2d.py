"""
Transform F2 and F1 in 2D NMR
================================

Transform a Bruker States acquisition of ²⁷Al PST-5 along F2 and F1,
then plot the real-real plane. The :doc:`MQMAS guide </userguide/mqmas>`
continues with phase correction and shearing. For the JEOL COSY workflow,
see :ref:`jeol-cosy-indirect-processing`.

Data: Christian Fernandez, LCS, distributed through
`spectrochempy_data <https://github.com/spectrochempy/spectrochempy_data>`_
under `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_.
"""

# %%
import spectrochempy as scp

# %%
# Correct the digital-filter delay at import and transform the direct axis.
bruker_fid = scp.nmr.read_topspin(
    "nmrdata/bruker/tests/nmr/MQMAS_27AL_PST5/51/ser", remove_digital_filter=True
)
bruker_f2 = bruker_fid.fft(dim="x")

# %%
# Keep both States branches for the indirect transform, then plot the
# real-real plane before MQMAS shearing. Both axes are in the frequency domain.
spectrum = bruker_f2.fft(dim="y")
rr = scp.nmr.Experiment(spectrum).rr()
region = rr[:, 65.0:20.0]
peak = region.max().magnitude
levels = [peak * fraction for fraction in (0.10, 0.16, 0.25, 0.40, 0.65, 0.90)]
_ = region.plot(
    method="contour", levels=levels, cmap="magma_r", title="PST-5 MQMAS before shearing"
)
