"""Public isotope lookup and magnetic-field conversion."""

from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest
import spectrochempy as scp


def test_isotope_properties():
    aluminium = scp.nmr.Isotope("<Al-27>")
    assert aluminium == scp.nmr.Isotope("27al") == scp.nmr.Isotope("Al27")
    assert aluminium.nucleus == "27Al"
    assert aluminium.spin == Fraction(5, 2)
    assert aluminium.abundance == 100
    assert aluminium.quadrupole_moment.to("barn").magnitude == pytest.approx(0.1466)
    with pytest.raises(FrozenInstanceError):
        aluminium.nucleus = "87Rb"
    with pytest.raises(ValueError, match="Unknown isotope"):
        scp.nmr.Isotope("99Xx")


def test_larmor_frequency_and_sign():
    proton = scp.nmr.Isotope("1H")
    nitrogen = scp.nmr.Isotope("15N")
    assert proton.gyromagnetic_ratio.to("MHz/T").magnitude == pytest.approx(
        42.5775, rel=1e-5
    )
    assert nitrogen.gyromagnetic_ratio.magnitude < 0
    assert nitrogen.larmor_frequency(9.4 * scp.ur.T).magnitude == pytest.approx(
        40.58, rel=1e-3
    )
    assert proton.larmor_frequency(0).magnitude == 0
    with pytest.raises(ValueError, match="finite and nonnegative"):
        proton.larmor_frequency(-1)
