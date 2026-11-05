"""Synthetic tests for the neutral interop-to-NDDataset converter."""

from types import SimpleNamespace

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import from_interop


def _normalized(values=None, *, axis_values=None, sequence="NOESY"):
    if values is None:
        values = np.asarray([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
    axis = SimpleNamespace(
        values=axis_values,
        unit="s" if axis_values is not None else None,
        label="H1",
        domain="time",
        direction="increasing",
        reference=SimpleNamespace(
            observation_frequency=125.0,
            observation_frequency_unit="MHz",
            offset=None,
            offset_unit=None,
            source="acquisition",
        ),
    )
    return SimpleNamespace(
        data=values,
        axes=(axis,),
        dimensions=("F2",),
        physical_dimensions=("F2",),
        domains=("time",),
        quadrature_mode="complex-1d",
        branch_order=("complex",),
        normalization_state="storage-normalized",
        normalization_profile="agilent-fid-1d-complex-v1",
        applied_operations=(
            {
                "name": "decode",
                "stage": "acquisition",
                "input_dimensions": ("storage",),
                "output_dimensions": ("F2",),
                "axis": "F2",
                "scale": None,
                "conjugation": "none",
                "sign": "none",
                "permutation": None,
                "evidence": "synthetic",
            },
        ),
        provenance={"vendor": "Agilent", "sequence": sequence},
        acquisition={"spectral_width_hz": 2000.0},
        metadata={"raw": "descriptive"},
        encoding=("QSIM",),
        diagnostics=(),
    )


def test_from_interop_converts_complex_fid_without_mutating_source():
    source = _normalized(axis_values=np.arange(3) / 2000.0)
    before = source.data.copy()

    result = from_interop(source)

    assert isinstance(result, scp.NDDataset)
    assert result.dims == ["x"]
    assert result.data.dtype == np.complex64
    np.testing.assert_array_equal(result.data, before)
    np.testing.assert_array_equal(result.coordset[0].data, np.arange(3) / 2000.0)
    assert result.coordset[0].units == scp.ur.s
    assert result.meta.normalization_state == "storage-normalized"
    assert result.meta.provenance["vendor"] == "Agilent"
    assert result.meta.acquisition["spectral_width_hz"] == 2000.0
    assert result.meta.source_metadata["raw"] == "descriptive"
    assert result.meta.interop_axes[0]["direction"] == "increasing"
    assert result.meta.interop_axes[0]["reference"]["observation_frequency"] == 125.0
    assert result.meta.applied_operations[0]["name"] == "decode"
    result.data[0] = 99 + 99j
    assert source.data[0] == before[0]


@pytest.mark.parametrize("sequence", ["NOESY", "COSY", "TOCSY", "synthetic"])
def test_from_interop_ignores_descriptive_sequence_identity(sequence):
    result = from_interop(_normalized(sequence=sequence))
    np.testing.assert_array_equal(result.data, [1 + 2j, 3 + 4j, 5 + 6j])


def test_from_interop_uses_index_axis_when_calibration_is_unknown():
    result = from_interop(_normalized())
    assert result.coordset[0].data.tolist() == [0, 1, 2]
    assert result.coordset[0].units is None


def test_from_interop_rejects_descriptive_result_and_invalid_contract():
    descriptive = SimpleNamespace(data=np.ones(3, dtype=complex))
    with pytest.raises(TypeError, match="normalized acquisition"):
        from_interop(descriptive)

    invalid = _normalized()
    invalid.domains = ("frequency",)
    with pytest.raises(ValueError, match="time domain"):
        from_interop(invalid)

    invalid = _normalized()
    invalid.applied_operations = ({},)
    with pytest.raises(ValueError, match="operation"):
        from_interop(invalid)
