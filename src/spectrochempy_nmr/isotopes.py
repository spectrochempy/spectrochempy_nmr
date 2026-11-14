"""Nuclear isotope properties from the table shipped with SpectroChemPy."""

import csv
import re
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from importlib.resources import files
from math import isfinite
from numbers import Real


@lru_cache(maxsize=1)
def _isotope_table():
    table = files("spectrochempy").joinpath("data", "databases", "isotopes.csv")
    with table.open(encoding="utf-8") as stream:
        return {
            row["isotope"].strip(): {
                key.strip(): value.strip() for key, value in row.items()
            }
            for row in csv.DictReader(stream)
        }


def _canonical_nucleus(nucleus):
    if not isinstance(nucleus, str):
        raise TypeError("An isotope label must be a string such as '27Al'")
    label = nucleus.strip().strip("<>").strip()
    mass_first = re.fullmatch(r"(\d+)([A-Za-z]{1,2})", label)
    if mass_first:
        mass, symbol = mass_first.groups()
    else:
        symbol_first = re.fullmatch(r"([A-Za-z]{1,2})-?(\d+)", label)
        if not symbol_first:
            raise ValueError(f"Unknown isotope {nucleus!r}")
        symbol, mass = symbol_first.groups()
    name = f"{int(mass)}{symbol.capitalize()}"
    if name not in _isotope_table():
        raise ValueError(f"Unknown isotope {nucleus!r}")
    return name


@dataclass(frozen=True)
class Isotope:
    """Physical properties of an isotope such as ²⁷Al.

    Parameters
    ----------
    nucleus : str
        Isotope label, such as "27Al", "Al27", or "Al-27".
    """

    nucleus: str

    def __post_init__(self):
        object.__setattr__(self, "nucleus", _canonical_nucleus(self.nucleus))

    @property
    def spin(self):
        """Nuclear spin as an exact fraction."""
        return Fraction(_isotope_table()[self.nucleus]["spin"])

    @property
    def abundance(self):
        """Natural abundance in percent."""
        return float(_isotope_table()[self.nucleus]["abundance"])

    @property
    def quadrupole_moment(self):
        """Electric quadrupole moment in barns."""
        from spectrochempy.core.units import ur

        return float(_isotope_table()[self.nucleus]["quadrupole"]) * ur.barn

    @property
    def gyromagnetic_ratio(self):
        """Signed gyromagnetic ratio divided by 2π, in MHz/T."""
        from spectrochempy.core.units import ur

        g = float(_isotope_table()[self.nucleus]["gn"])
        return (g * ur.nuclear_magneton / ur.planck_constant).to("MHz/T")

    def larmor_frequency(self, field):
        """Return the nonnegative Larmor frequency in MHz for a field in tesla.

        A numerical field is interpreted in tesla. A SpectroChemPy quantity
        with magnetic field units is also accepted.
        """
        from spectrochempy.core.units import ur

        if isinstance(field, ur.Quantity):
            field = field.to("T").magnitude
        if isinstance(field, bool) or not isinstance(field, Real):
            raise TypeError("Magnetic field must be a scalar in tesla")
        if not isfinite(field) or field < 0:
            raise ValueError("Magnetic field must be finite and nonnegative")
        return (abs(self.gyromagnetic_ratio) * (field * ur.T)).to("MHz")


def nuclear_spin(nucleus):
    """Return the exact spin of a known isotope, or None."""
    try:
        return Isotope(nucleus).spin
    except (TypeError, ValueError):
        return None
