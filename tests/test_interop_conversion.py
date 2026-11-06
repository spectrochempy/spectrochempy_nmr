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


def test_from_interop_preserves_small_dwell_on_long_acquisition():
    axis_values = np.arange(32768) / 12019.2307692
    source = _normalized(axis_values=axis_values)
    source.data = np.ones(len(axis_values), dtype=np.complex64)
    result = from_interop(source)
    np.testing.assert_allclose(result.x.data, axis_values, rtol=1e-10, atol=1e-14)
    assert np.all(np.diff(result.x.data) > 0)


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


def test_from_interop_rejects_masked_samples_instead_of_dropping_the_mask():
    source = _normalized(
        np.ma.array([1 + 2j, 3 + 4j, 5 + 6j], mask=[False, True, False])
    )

    with pytest.raises(ValueError, match="unmasked"):
        from_interop(source)


@pytest.mark.parametrize(
    "values",
    [
        np.arange(2),
        np.arange(3).reshape(1, 3),
        [0, np.nan, 2],
        [0, np.inf, 2],
        [0 + 1j, 1 + 1j, 2 + 1j],
        ["0", "1", "2"],
    ],
)
def test_from_interop_rejects_invalid_time_coordinates(values):
    with pytest.raises(ValueError, match="axis"):
        from_interop(_normalized(axis_values=values))


def test_from_interop_rejects_frequency_axis_on_time_domain_data():
    source = _normalized(axis_values=np.arange(3))
    source.axes[0].domain = "frequency"

    with pytest.raises(ValueError, match="axis"):
        from_interop(source)


def test_from_interop_detaches_nested_array_metadata():
    source = _normalized(axis_values=np.arange(3) / 2000.0)
    source.metadata = {"nested": {"coefficients": np.array([1.0, 2.0])}}
    source.provenance["checksums"] = np.array([1, 2])
    source.diagnostics = (
        SimpleNamespace(
            code="note",
            severity="info",
            message="test",
            context={"values": np.array([3.0, 4.0])},
        ),
    )

    result = from_interop(source)

    result.meta.source_metadata["nested"]["coefficients"][0] = 99
    result.meta.provenance["checksums"][0] = 99
    result.meta.diagnostics[0]["context"]["values"][0] = 99
    result.coordset[0].data[1] = 99
    np.testing.assert_array_equal(source.metadata["nested"]["coefficients"], [1, 2])
    np.testing.assert_array_equal(source.provenance["checksums"], [1, 2])
    np.testing.assert_array_equal(source.diagnostics[0].context["values"], [3, 4])
    np.testing.assert_array_equal(source.axes[0].values, np.arange(3) / 2000.0)


def test_from_interop_accepts_an_unmasked_array_and_unknown_axis():
    source = _normalized(np.ma.array([1 + 2j, 3 + 4j], mask=False))
    source.axes[0].direction = "unknown"

    result = from_interop(source)

    np.testing.assert_array_equal(result.data, source.data.data)
    np.testing.assert_array_equal(result.coordset[0].data, [0, 1])
    assert result.coordset[0].units is None
    assert result.meta.interop_axes[0]["direction"] == "unknown"


def test_from_interop_rejects_masked_coordinates():
    source = _normalized(
        axis_values=np.ma.array([0.0, 0.1, 0.2], mask=[False, True, False])
    )

    with pytest.raises(ValueError, match="axis.*unmasked"):
        from_interop(source)


@pytest.mark.parametrize("unit", ["Hz", "ppm", "m"])
@pytest.mark.parametrize("axis_values", [None, np.arange(3)])
def test_from_interop_rejects_non_time_units_for_time_coordinates(unit, axis_values):
    source = _normalized(axis_values=axis_values)
    source.axes[0].unit = unit

    with pytest.raises(ValueError, match="axis units"):
        from_interop(source)


@pytest.mark.parametrize("unit", ["s", "us"])
def test_from_interop_preserves_explicit_time_units(unit):
    source = _normalized(axis_values=np.arange(3))
    source.axes[0].unit = unit

    result = from_interop(source)

    assert result.coordset[0].units == scp.ur.Unit(unit)
    np.testing.assert_array_equal(result.coordset[0].data, source.axes[0].values)
