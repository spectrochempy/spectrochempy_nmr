"""
Find and fit peaks in a 1D NMR spectrum
=======================================

Estimate the baseline, locate the main resonances and fit overlapping
peaks in a real proton spectrum using SpectroChemPy.
"""

# %%
import spectrochempy as scp

fid = scp.nmr.read("nmrdata/bruker/tests/nmr/topspin_1d/1/fid")
spectrum = scp.nmr.Experiment(fid).process(
    apodization="em", lb=2.0, size=16384, digital_filter="time"
)

# %%
# Work on the real, baseline-corrected spectrum. Specify the minimum peak
# separation in samples, then use the detected sample indices to get the
# positions in ppm from the original coordinate.
corrected = spectrum.real[-7.0:12.0].snip()
result = corrected.find_peaks(
    height=50000.0, distance=20, use_coord=False, as_result=True
)
markers = corrected[result.sample_index]
result

# %%
ax = corrected.plot()
_ = markers.plot_scatter(ax=ax, clear=False, data_only=True, marker="v", color="black")

# %%
# Integrate the dominant resonance on the calibrated ppm axis. The bounds
# select this peak for this dataset.
area = corrected[-0.1:1.5].trapezoid()
area

# %%
# Initialize one Voigt profile per detected peak. Keep the positions close
# to their starting values and share the width and mixing parameter.
models = ["COMMON:\n$ commonwidth: 0.2, 0.01, 1.0\n$ commonratio: 0.5, 0, 1"]
for i, (position, height) in enumerate(zip(markers.x.data, markers.data, strict=True)):
    models.append(
        f"MODEL: peak{i + 1}\n"
        "shape: voigtmodel\n"
        f"$ ampl: {height:.1f}, 0, none\n"
        f"$ pos: {position:.4f}, {position - 0.25:.4f}, {position + 0.25:.4f}\n"
        "> ratio: commonratio\n"
        "> width: commonwidth"
    )

fitter = scp.Optimize()
fitter.script = "\n\n".join(models)
_ = fitter.fit(corrected)

# %%
# Compare the fitted profile with the observed spectrum and inspect the residual.
_ = fitter.plot_merit()
