"""Optional public-reader integration with the private historical MQMAS map.

This checks integration and ownership, not TopSpin xfshear numerical parity.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import Experiment


def test_mqmas_public_processed_reader_preserves_input():
    path = os.environ.get("SCP_MQMAS_PDATA")
    if not path:
        pytest.skip("set SCP_MQMAS_PDATA to the historical processed 27Al MQMAS 2rr")
    assert Path(path).is_file(), "SCP_MQMAS_PDATA must name an existing 2rr file"
    source = scp.nmr.read_topspin(path, local_only=True, proc_axis=True)
    original_data = source.data.copy()
    original_x = source.x.data.copy()
    original_y = source.y.data.copy()
    original_history = list(source.history)
    result = Experiment(source).shear_mqmas(spin=2.5)
    assert result is not source
    assert result.dims == source.dims
    assert result.shape == source.shape
    assert result.units == source.units
    assert np.isfinite(result.data).all()
    np.testing.assert_array_equal(source.data, original_data)
    np.testing.assert_array_equal(source.x.data, original_x)
    np.testing.assert_array_equal(source.y.data, original_y)
    assert source.history == original_history
    np.testing.assert_allclose(result.x.data, source.x.data, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        result.y.to("Hz").data, source.y.to("Hz").data / (17 / 12), rtol=1e-12
    )
    assert result.meta.component_labels == source.meta.component_labels
    assert not np.shares_memory(result.data, source.data)


@pytest.mark.parametrize(
    ("sx", "reference_env"),
    [(2048, "SCP_MQMAS_XFSHEAR_PDATA"), (4096, "SCP_MQMAS_XFSHEAR_ZF_PDATA")],
)
def test_replay_xfshear_matches_mqmas_51_four_planes(sx, reference_env):
    ser = os.environ.get("SCP_MQMAS_SER")
    pdata = os.environ.get(reference_env)
    if not ser or not pdata:
        pytest.skip("requires MQMAS/51 SER and its rectangular XFSHEAR reference")
    source = scp.nmr.read_topspin(
        ser, local_only=True, remove_digital_filter=False, remove_dc_offset=False
    )
    original = source.data.copy()
    ref = scp.nmr.read_topspin(pdata, local_only=True, proc_axis=True)
    result = Experiment(source).replay_xfshear(
        spin=2.5, size=(512, sx), direct_points=2048
    )
    np.testing.assert_array_equal(source.data, original)
    np.testing.assert_allclose(result.x.data, ref.x.data, rtol=0, atol=1e-7)
    np.testing.assert_allclose(result.y.data, ref.y.data, rtol=0, atol=1e-7)

    # One amplitude factor for all four planes.
    rr = result.data[0].real
    gain = np.vdot(rr, ref.data[0].real) / np.vdot(rr, rr)
    assert gain == pytest.approx(7.5, rel=2e-6)
    for plane, channel, part, tolerance in (
        ("RR", 0, "real", 1e-3),
        ("RI", 1, "real", 2e-6),
        ("IR", 0, "imag", 2e-6),
        ("II", 1, "imag", 2e-6),
    ):
        expected = getattr(ref.data[channel], part)
        actual = getattr(result.data[channel], part) * gain
        assert (
            np.linalg.norm(actual - expected) / np.linalg.norm(expected) < tolerance
        ), plane

    corrected = scp.nmr.read_topspin(ser, local_only=True, remove_dc_offset=True)
    with pytest.raises(ValueError, match="uncorrected"):
        Experiment(corrected).replay_xfshear(spin=2.5, size=(512, 2048))
