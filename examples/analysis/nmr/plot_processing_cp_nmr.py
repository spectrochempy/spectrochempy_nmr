"""
Follow cross-polarization buildup in a Bruker series
====================================================

Compare how three regions of a solid-state carbon-13 spectrum respond
to the cross-polarization (CP) contact time. The Bruker experiment stores
each of the 15 FIDs in a separate experiment directory.
"""

# %%
# sphinx_gallery_thumbnail_number = 2

import spectrochempy as scp

root = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/CP"
fids = [scp.nmr.read_topspin(root / str(expno) / "fid") for expno in range(80, 95)]


# %%
# Read P15 from each acquisition. In this series, P15 is the contact
# pulse duration. Sort the FIDs by that duration before stacking them.
def contact_ms(fid):
    return fid.meta.p[0][15].to("ms").magnitude


fids.sort(key=contact_ms)
contact_time = scp.Coord(
    [contact_ms(fid) for fid in fids], units="ms", title="CP contact time"
)


# %%
# These acquisitions share a direct-axis grid. Build a pseudo-2D dataset
# with one row per contact time. The contact axis is a series coordinate.
def contact_series(rows):
    return scp.NDDataset(
        [row.data for row in rows],
        coordset=[contact_time, rows[0].x],
        units=rows[0].units,
        title="signal",
    )


series = contact_series(fids)
series

# %%
# Process each FID on its own direct axis so it keeps its acquisition
# metadata through the FFT. Use magnitude spectra for this comparison
# of peak heights across contact times.
spectra = contact_series([abs(fid.em(lb=50.0).fft(size=4096)) for fid in fids])
_ = spectra.plot(method="lines", xlim=(190.0, 50.0), colorbar=True)

# %%
# Follow the maximum in each selected ppm window. These curves illustrate
# the buildup and subsequent decay of the signal in different regions.
windows = [(165.0, 183.0), (91.0, 107.0), (62.0, 79.0)]
curves = [spectra[:, low:high].max(dim="x") for low, high in windows]
_ = scp.plot_multiple(
    curves,
    method="scatter",
    labels=[f"{low:g}–{high:g} ppm" for low, high in windows],
    title="Carbon-13 CP buildup",
    legend="upper right",
    show=False,
)
