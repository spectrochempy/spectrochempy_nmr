"""Access the nuclear-spin data shipped with SpectroChemPy."""

import csv
from fractions import Fraction
from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=1)
def _isotope_spins():
    """Read the isotope table shipped with SpectroChemPy."""
    table = files("spectrochempy").joinpath("data", "databases", "isotopes.csv")
    with table.open(encoding="utf-8") as stream:
        return {
            row["isotope"].strip(): Fraction(row["spin"].strip())
            for row in csv.DictReader(stream)
        }


def nuclear_spin(nucleus):
    """Return the exact spin of a known isotope, or None."""
    return _isotope_spins().get(nucleus)
