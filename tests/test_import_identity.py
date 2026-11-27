"""Automatic import assembly compares plugin-owned physical NMR facts."""

import numpy as np
import pytest
import spectrochempy as scp
from spectrochempy.core.readers.importer import merge_datasets

from spectrochempy_nmr import NMRPlugin, from_interop
from spectrochempy_nmr.import_identity import _measurement_identity
from test_interop_conversion import _normalized
from test_processed_interop import _real_plane, _spectrum


@pytest.fixture(autouse=True)
def identity_handler(monkeypatch):
    from spectrochempy.core.readers import importer
    from spectrochempy.plugins import manager

    if not hasattr(importer, "_import_identities"):
        pytest.skip("requires the Core importer.measurement_identity extension")

    registry = manager.plugin_manager.registry
    original = registry.get_handler
    handler = NMRPlugin().register_handlers()["importer.measurement_identity"]
    monkeypatch.setattr(
        registry,
        "get_handler",
        lambda name: handler
        if name == "importer.measurement_identity"
        else original(name),
    )


def fid(nucleus="1H", encoding="QSIM"):
    source = _normalized(axis_values=np.arange(3) / 2000)
    source.acquisition["nucleus"] = nucleus
    source.encoding = (encoding,)
    return from_interop(source)


@pytest.mark.parametrize("nucleus", ["H1", "H-1", "<1H>", "1H"])
def test_equivalent_nuclei_and_decoded_encoding_merge(nucleus):
    first, second = fid(), fid(nucleus, "DQD")
    result = merge_datasets([first, second], merge=True, sortbydate=False)
    assert len(result) == 1
    assert result[0].shape == (2, 3)
    np.testing.assert_array_equal(result[0].data, [first.data, second.data])


@pytest.mark.parametrize("explicit", [False, True])
def test_conflicting_nuclei_do_not_merge(explicit):
    first, second = fid(), fid("13C")
    before = second.copy()
    with pytest.warns(UserWarning, match="nmr.F2.nucleus differ"):
        result = merge_datasets(
            [first, second], **({"merge": True} if explicit else {})
        )
    assert len(result) == 2
    np.testing.assert_array_equal(second.data, before.data)
    np.testing.assert_array_equal(second.x.data, before.x.data)


@pytest.mark.parametrize("unknown", [None, "unidentified"])
def test_unknown_nucleus_does_not_invent_a_conflict(unknown):
    result = merge_datasets([fid(unknown), fid("13C")], merge=True, sortbydate=False)
    assert len(result) == 1


@pytest.mark.parametrize("field", ["domain", "representation"])
def test_known_domain_or_representation_contradiction(field):
    first, second = fid(), fid()
    if field == "domain":
        second.meta.data_domain = "frequency"
    else:
        second = second.real
    with pytest.warns(UserWarning, match=f"nmr.F2.{field} differ"):
        assert len(merge_datasets([first, second], merge=True)) == 2


def test_unknown_cannot_bridge_nuclei_after_intermediate_assembly():
    first = merge_datasets([fid(None), fid()], merge=True, sortbydate=False)[0]
    second = merge_datasets([fid("13C"), fid("13C")], merge=True, sortbydate=False)[0]
    with pytest.warns(UserWarning, match="nmr.F2.nucleus differ"):
        result = merge_datasets([first, second], merge=True)
    assert len(result) == 2
    assert [ds.shape for ds in result] == [(2, 3), (2, 3)]


@pytest.mark.parametrize("route", ["paths", "directory"])
def test_public_native_import_applies_nmr_constraints(tmp_path, route):
    paths = [
        dataset.dump(tmp_path / f"{index}.scp")
        for index, dataset in enumerate([fid(None), fid(), fid("13C"), fid("13C")])
    ]
    with pytest.warns(UserWarning, match="nmr.F2.nucleus differ"):
        result = (
            scp.read(paths, merge=True, sortbydate=False)
            if route == "paths"
            else scp.read_dir(tmp_path, merge=True, sortbydate=False)
        )
    assert len(result) == 2
    assert sum(ds.shape[0] for ds in result) == 4
    assert all("_import_measurement_identity" not in ds.__dict__ for ds in result)


@pytest.mark.parametrize("native", [False, True])
def test_physical_2d_needs_a_new_series_axis(tmp_path, native):
    first, second = from_interop(_real_plane()), from_interop(_real_plane())
    before = first.data.copy()
    with pytest.warns(UserWarning, match="physical multidimensional NMR"):
        if native:
            paths = [
                ds.dump(tmp_path / f"{i}.scp") for i, ds in enumerate([first, second])
            ]
            result = scp.read(paths, merge=True)
        else:
            result = merge_datasets([first, second], merge=True)
    assert len(result) == 2
    assert all(ds.shape == (3, 8) for ds in result)
    np.testing.assert_array_equal(result[0].data, before)


def test_processed_real_and_complex_representations_are_distinct():
    first, second = (
        from_interop(_spectrum(complex_data=value)) for value in [False, True]
    )
    with pytest.warns(UserWarning, match="nmr.F2.representation differ"):
        assert len(merge_datasets([first, second], merge=True)) == 2


def test_non_nmr_and_vendor_name_alone_do_not_supply_nmr_facts():
    dataset = scp.NDDataset([1, 2, 3], origin="topspin", title="1H NMR")
    assert _measurement_identity(dataset) is None
    dataset.meta.origin = "topspin"
    assert _measurement_identity(dataset) is None


@pytest.mark.parametrize("complex_data", [False, True])
def test_legacy_topspin_metadata_supplies_normalized_facts(complex_data):
    dataset = scp.NDDataset([1 + 1j, 2 + 2j, 3 + 3j])
    if not complex_data:
        dataset = dataset.real
    dataset.meta.update(
        {
            "origin": "topspin",
            "isfreq": [False],
            "iscomplex": [True],
            "nuc1": ["H1"],
            "encoding": [3],
        }
    )
    facts = _measurement_identity(dataset)["constraints"]
    assert facts["nmr.F2.nucleus"] == "1H"
    assert facts["nmr.F2.domain"] == "time"
    assert facts["nmr.F2.representation"] == ("complex" if complex_data else "real")


def test_legacy_unknown_domain_is_not_inferred_as_time():
    dataset = scp.NDDataset([1, 2, 3])
    dataset.meta.update({"origin": "topspin", "isfreq": [None], "nuc1": ["1H"]})
    assert "nmr.F2.domain" not in _measurement_identity(dataset)["constraints"]


def test_transposed_1d_series_is_not_concatenated_along_the_spectral_axis():
    series = merge_datasets([fid(), fid()], merge=True, sortbydate=False)[0].T
    with pytest.warns(UserWarning, match="observation axis first"):
        assert len(merge_datasets([series, series.copy()], merge=True)) == 2


def test_public_simpson_reader_still_merges_compatible_traces(tmp_path):
    paths = []
    for i in range(2):
        path = tmp_path / f"trace-{i}.spe"
        path.write_text(
            "SIMP\nNP=4\nSW=10000\nTYPE=SPE\nDATA\n1 0\n0.5 0.25\n0 0.5\n-0.25 0\nEND\n"
        )
        paths.append(path)
    datasets = scp.nmr.read_simpson(paths, merge=True, sortbydate=False)
    assert isinstance(datasets, scp.NDDataset)
    assert datasets.shape == (2, 4)
    assert "_import_measurement_identity" not in datasets.__dict__


def series(nucleus="1H"):
    dataset = scp.NDDataset(
        np.ones((2, 3), dtype=complex),
        coordset=[scp.Coord([0, 1]), scp.Coord([0, 0.0005, 0.001], units="s")],
    )
    dataset.meta.update(
        {
            "interop_dimensions": ("trace", "F2"),
            "interop_physical_dimensions": ("trace", "F2"),
            "axis_roles": ("trace", "F2"),
            "quadrature_mode": "series-stored-2d",
            "normalization_state": "storage-normalized",
            "data_domain": "time",
            "acquisition": {},
            "td": [2, 3],
            "isfreq": [False, False],
            "iscomplex": [False, True],
            "nucleus": [None, nucleus],
            "encoding": ["undefined", "QSIM"],
        }
    )
    return dataset


def test_pseudo2d_observation_series_can_merge():
    first, second = series(), series("H1")
    result = merge_datasets([first, second], merge=True, sortbydate=False)
    assert len(result) == 1
    assert result[0].shape == (4, 3)
    assert _measurement_identity(first)["constraints"]["nmr.F2.nucleus"] == "1H"


def test_pseudo2d_series_still_checks_direct_nucleus():
    with pytest.warns(UserWarning, match="nmr.F2.nucleus differ"):
        assert len(merge_datasets([series(), series("13C")], merge=True)) == 2


@pytest.mark.parametrize("case_index", range(4))
def test_available_native_reader_supplies_nmr_facts(tmp_path, case_index):
    from test_reader_metadata_options import _reader_cases

    reader, path, vendor = _reader_cases(tmp_path)[case_index]
    if not path.exists():
        pytest.skip(f"optional local {vendor} acquisition unavailable")
    dataset = reader(path)
    descriptor = _measurement_identity(dataset)
    assert descriptor is not None
    assert descriptor["reason"] is None
    assert descriptor["constraints"]["nmr.F2.domain"] in {"time", "frequency"}


def test_unrelated_collection_metadata_is_not_claimed_by_nmr():
    dataset = scp.NDDataset([1, 2, 3])
    dataset.meta.origin = ["optical", "topspin"]
    assert _measurement_identity(dataset) is None


@pytest.mark.parametrize("quadrature", ["states", "states-tppi"])
@pytest.mark.parametrize("component", [False, True])
def test_public_paired_acquisitions_keep_physical_channels_separate(
    tmp_path, quadrature, component
):
    from spectrochempy_nmr import read
    from test_jcampdx_states import _write

    paths = []
    for index in range(2):
        path = tmp_path / f"paired-{index}.dx"
        _write(path, np.ones((4, 3), dtype=complex), quadrature)
        paths.append(path)
    first, second = [
        read(
            path,
            protocol="interop",
            profile="jcampdx-fid-2d",
            conversion_profile=quadrature,
            normalize_options={
                "quadrature": quadrature,
                "conjugation": "none",
                "confirm_adjacent_pairs": True,
                "indirect_dwell_s": 1 / 3200,
            },
        )
        for path in paths
    ]
    assert first.shape == (2, 2, 3)
    if component:
        first, second = first[0].squeeze(), second[0].squeeze()
    expected_reason = (
        "current NMR dimensions differ"
        if component
        else "physical multidimensional NMR"
    )
    saved = [
        ds.dump(tmp_path / f"paired-{i}.scp") for i, ds in enumerate([first, second])
    ]
    with pytest.warns(UserWarning, match=expected_reason):
        result = scp.read(saved, merge=True)
    assert len(result) == 2
    for actual, expected in zip(result, [first, second], strict=True):
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual.data, expected.data)
        assert tuple(actual.meta.axis_roles) == tuple(expected.meta.axis_roles)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("frequency", [False, True])
def test_public_real_component_uses_current_representation(tmp_path, native, frequency):
    first = from_interop(_spectrum()) if frequency else fid()
    second = first.real
    assert second.meta.quadrature_mode == "complex-1d"
    assert not second.is_complex
    assert (
        _measurement_identity(second)["constraints"]["nmr.F2.representation"] == "real"
    )
    with pytest.warns(UserWarning, match="nmr.F2.representation differ"):
        if native:
            paths = [
                ds.dump(tmp_path / f"{i}.scp") for i, ds in enumerate([first, second])
            ]
            result = scp.read(paths, merge=True, sortbydate=False)
        else:
            result = merge_datasets([first, second], merge=True, sortbydate=False)
    assert len(result) == 2
    assert sorted(ds.is_complex for ds in result) == [False, True]
    for dataset in result:
        expected = first if dataset.is_complex else second
        np.testing.assert_array_equal(dataset.data, expected.data)


def test_real_observation_series_still_merge_with_other_real_series():
    real_series = series().real
    other = real_series.copy()
    assert (
        _measurement_identity(real_series)["constraints"]["nmr.F2.representation"]
        == "real"
    )
    merged = merge_datasets([real_series, other], merge=True, sortbydate=False)
    assert len(merged) == 1
    assert merged[0].shape == (4, 3)
    assert not merged[0].is_complex
    with pytest.warns(UserWarning, match="nmr.F2.representation differ"):
        assert len(merge_datasets([series(), real_series], merge=True)) == 2


@pytest.mark.parametrize("merge", [None, True, False])
@pytest.mark.parametrize("axis", ["row", "column"])
def test_sliced_native_inputs_remain_readable_when_acquisition_axes_are_stale(
    tmp_path, merge, axis
):
    plane = from_interop(_real_plane())
    dataset = (plane[0] if axis == "row" else plane[:, 0]).squeeze()
    assert dataset.ndim == 1
    paths = [dataset.dump(tmp_path / f"slice-{i}.scp") for i in range(2)]
    options = {} if merge is None else {"merge": merge}
    if merge is False:
        result = scp.read(paths, **options)
    else:
        with pytest.warns(UserWarning, match="current NMR dimensions differ"):
            result = scp.read(paths, **options)
    assert len(result) == 2
    for restored in result:
        assert restored.shape == dataset.shape
        assert restored.dims == dataset.dims
        assert restored.units == dataset.units
        assert restored.coordset == dataset.coordset
        np.testing.assert_array_equal(restored.data, dataset.data)
