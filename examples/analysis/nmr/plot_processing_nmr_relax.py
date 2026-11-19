"""
Fit four components across a recovery series
=============================================

Process F2 and fit four Voigt profiles with saturation-recovery amplitudes.
The detailed :doc:`/userguide/relaxation_t1` explains processing, bounds,
shared parameters and residuals. This recipe requires the development Core
API from `PR #1905 <https://github.com/spectrochempy/spectrochempy/pull/1905>`_;
the documentation environment pins its tested revision.
"""

import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/relax/100/ser"
raw = scp.nmr.read(path, use_list="vdlist", remove_digital_filter=True)
f2 = raw.em(lb=15 * scp.ur.Hz, dim="x").fft(dim="x")
phase = scp.nmr.apk(f2[-1].squeeze()[-140.0:-80.0], order=0, apply=False)
if not phase.estimate.success:
    raise RuntimeError(phase.estimate.message)
X = f2.pk(phc0=phase.estimate.p0 * scp.ur.deg, phc1=0 * scp.ur.deg)[
    :, -120.0:-85.0
].real
peaks = [
    {
        "height": {"value": amplitude, "bounds": (0, None)},
        "axes": {
            "x": {
                "model": "voigt",
                "parameters": {
                    "center": {"value": center, "bounds": (center - 2, center + 2)},
                    "width": {"value": 4, "bounds": (0.5, 12)},
                    "ratio": {"shared": "mixing"},
                },
            },
            "y": {
                "model": "saturation_recovery",
                "parameters": {
                    "tau": {"value": 2.3, "bounds": (0.1, 100)},
                },
            },
        },
    }
    for center, amplitude in [(-91.5, 50), (-97.0, 290), (-103.0, 350), (-108.5, 115)]
]
result = (
    scp.Optimize(
        peaks=peaks,
        shared_parameters={
            "mixing": {"value": 0.8, "bounds": (0, 1)},
        },
    )
    .fit(X)
    .result
)
print("Converged:", result.diagnostics["success"])
print(
    "Recovery times / s:",
    [round(p["axes"]["y"]["parameters"]["tau"], 3) for p in result.peaks],
)
_ = scp.plot_multiple(
    [
        X[-1].squeeze(),
        result.fitted[-1].squeeze(),
        *[c[-1].squeeze() for c in result.components],
    ],
    labels=["Measured", "Global fit", "Band 1", "Band 2", "Band 3", "Band 4"],
    method="pen",
    legend="best",
    title="Four-component recovery fit",
)
