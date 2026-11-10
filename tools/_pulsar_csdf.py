"""Read the real linear CSDF format used by the DMFIT qualifications."""

import base64
import json

import numpy as np


def read_real_csdf(path):
    """Read the real, linear frequency data in a DMFIT CSDF."""
    data = json.loads(path.read_text(encoding="utf-8"))["csdm"]
    dimension = data["dimensions"][0]
    variable = data["dependent_variables"][0]
    if (
        dimension["type"] != "linear"
        or variable["numeric_type"] != "float32"
        or variable["encoding"] != "base64"
    ):
        raise ValueError(
            "Expected a real base64 float32 spectrum with a linear frequency axis."
        )
    increment, unit = dimension["increment"].split()
    offset, offset_unit = dimension["coordinates_offset"].split()
    larmor, larmor_unit = dimension["origin_offset"].split()
    if (unit, offset_unit, larmor_unit) != ("Hz", "Hz", "MHz"):
        raise ValueError("Expected Hz for the axis and MHz for the Larmor frequency.")
    values = np.frombuffer(base64.b64decode(variable["components"][0]), dtype="<f4")
    count = dimension["count"]
    if values.size != count or float(increment) >= 0:
        raise ValueError("Expected the complete spectrum with a decreasing axis.")
    frequency = float(offset) + np.arange(count) * float(increment)
    return frequency[::-1], values[::-1].astype(float), float(larmor) * 1e6
