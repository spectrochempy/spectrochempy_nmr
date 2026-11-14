"""Exercise phase selection and the Bruker storage boundary through the public API."""

import numpy as np
import pytest
import spectrochempy as scp

from nmrglue import bruker


@pytest.fixture
def spam_ser(tmp_path):
    """Two asymmetric phase harmonics, with complex and unequal SPAM scans."""
    a = np.array([[32 + 14j, 51 - 9j], [15 - 31j, -10 + 4j]])
    b = np.array([[8 + 3j, -4 + 11j], [5 - 2j, 9 + 10j]])
    theta = 2 * np.pi * np.arange(11) / 11
    plus = 2 * (
        a[:, None, :] * np.exp(-3j * theta[None, :, None])
        + b[:, None, :] * np.exp(-5j * theta[None, :, None])
    )
    minus = 2 * b[:, None, :] * np.exp(3j * theta[None, :, None])
    traces = np.stack(((plus + minus) / 2, (plus - minus) / 2), axis=2)
    (tmp_path / "pdata/1").mkdir(parents=True)
    for name, params in {
        "acqus": {
            "PULPROG": "MQMultiplex52",
            "PARMODE": 2,
            "AQ_mod": 1,
            "TD": 4,
            "IN": [0.0001, 0.0002],
            "SW_h": 50000.0,
            "SFO1": 104.3,
            "NUC1": "27Al",
            "GRPDLY": 2.0,
            "BYTORDA": 1,
            "DTYPA": 2,
        },
        "acqu2s": {"TD": 2},
        "acqu3s": {"TD": 22},
        "pdata/1/procs": {"SF": 104.29},
    }.items():
        bruker.write_jcamp(
            {"_coreheader": [], "_comments": [], **params}, tmp_path / name
        )
    # Float64 storage with a 1024-byte stride. Nonzero padding words
    # distinguish the physical complex points from the stored row length.
    words = np.ones((45, 128), dtype=">f8")
    words[:44, :4:2] = traces.reshape(44, 2).real
    words[:44, 1:4:2] = traces.reshape(44, 2).imag
    words[44, :4] = 2
    words.tofile(tmp_path / "ser")
    return tmp_path, a, b


def test_msm_branches_and_coherence_alias_guard(spam_ser):
    path, a, b = spam_ser
    read = scp.nmr.read_multiplex_spam
    before = (path / "ser").read_bytes()
    with pytest.raises(ValueError, match="Unexpected extra SER data"):
        read(path)
    with pytest.raises(ValueError, match="Unsupported MQ coherence"):
        read(path, coherence_order=7, allow_partial=True)

    three = read(path / "ser", coherence_order=3, allow_partial=True)
    five = read(path, coherence_order=5, allow_partial=True)
    np.testing.assert_allclose(three.data, np.stack((b - a, 1j * (a + b))))
    np.testing.assert_allclose(five.data, np.stack((-b, 1j * b)))
    np.testing.assert_allclose(three.y.to("s").data, [0, 0.0001])
    np.testing.assert_allclose(three.x.to("s").data, [0, 0.00002])
    assert tuple(three.c.labels) == ("E1", "E2")
    assert three.meta.acquisition["multiplex_spam"]["ignored_nonzero_traces"] == 1

    xf2 = scp.nmr.Experiment(three).xf2(size=8)
    phased = scp.nmr.Experiment(xf2).xf2p()
    with pytest.raises(ValueError, match="must match"):
        scp.nmr.Experiment(phased).xfshear(size=4, spin=2.5, coherence_order=5)
    spectrum = scp.nmr.Experiment(phased).xfshear(size=4, spin=2.5, coherence_order=3)
    assert spectrum.shape == (2, 4, 8)
    assert (path / "ser").read_bytes() == before
