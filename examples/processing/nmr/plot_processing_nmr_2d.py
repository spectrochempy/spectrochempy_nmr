"""
Transform F2 and F1 in 2D NMR
================================

Process the direct dimension of the epicatechin COSY acquisition from
`nmrXiv project P33 <https://doi.org/10.57992/nmrxiv.p33>`_ (CC0 1.0).
The result has a frequency axis in F2 and retains the indirect acquisition
time axis in F1. For this JEOL profile, processing stops after F2 because
the indirect quadrature convention has yet to be established.

Then use a Bruker States acquisition of ²⁷Al PST-5 to transform both
dimensions and plot the real-real plane. The :doc:`MQMAS guide
</userguide/mqmas>` continues with phase correction and shearing.
"""

# %%
import spectrochempy as scp
# sphinx_gallery_thumbnail_number = 2

path = scp.preferences.datadir / "nmrdata/jeol/epicatechin_cosy.jdf"
fid = scp.nmr.read_jeol(path)

# %%
# Apply an exponential window and Fourier-transform the direct axis.
windowed = scp.em(fid, lb=2.0, dim="x")
mixed = scp.fft(windowed, dim="x")
mixed

# %%
# Overlay the real F2 spectra for the first five F1 increments.
traces: scp.NDDataset = mixed[:5, :].real
_ = traces.plot(method="lines")

# %%
# The Bruker acquisition has a supported States convention. Correct the
# digital-filter delay at import, transform the direct axis and retain both
# States branches for the indirect transform. Plot the real-real plane
# before MQMAS shearing. Both axes are now in the frequency domain.
bruker_fid = scp.nmr.read_topspin(
    "nmrdata/bruker/tests/nmr/MQMAS_27AL_PST5/51/ser", remove_digital_filter=True
)
bruker_f2 = bruker_fid.fft(dim="x")
spectrum = bruker_f2.fft(dim="y")
rr = scp.nmr.Experiment(spectrum).rr()
region = rr[:, 65.0:20.0]
peak = region.max().magnitude
levels = [peak * fraction for fraction in (0.10, 0.16, 0.25, 0.40, 0.65, 0.90)]
_ = region.plot(
    method="contour", levels=levels, cmap="magma_r", title="PST-5 MQMAS before shearing"
)
