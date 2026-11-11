"""Numerical conventions and sequencing for MQMAS XFSHEAR."""

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import Experiment
from spectrochempy_nmr.processing.mqmas import _mqmas_geometry


@pytest.mark.parametrize(
    ("spin", "order", "ratio"),
    [
        (1.5, 3, -7 / 9),
        (2.5, 3, 19 / 12),
        (2.5, 5, -25 / 12),
        (3.5, 3, 101 / 45),
        (4.5, 9, -31 / 6),
    ],
)
def test_xfshear_signed_ratios(spin, order, ratio):
    _, actual, scale = _mqmas_geometry(spin, order)
    assert actual == pytest.approx(ratio)
    assert scale == pytest.approx(order - ratio)


@pytest.mark.parametrize(
    "spin,order",
    [(1, 3), (2.1, 3), (True, 3), (np.nan, 3), (2.5, 2), (2.5, 7), (2.5, 3.0)],
)
def test_xfshear_rejects_invalid_transition(spin, order):
    with pytest.raises(ValueError):
        _mqmas_geometry(spin, order)


def test_xfshear_requires_the_xf2_stage():
    spectrum = scp.NDDataset(np.ones((8, 8)))
    with pytest.raises(ValueError, match="prepare_xfshear"):
        Experiment(spectrum).replay_xfshear(spin=2.5)
