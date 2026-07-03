"""Tests for the EvoForestTSWM transformer."""

import numpy as np
import pytest

from aeon.transformations.collection.feature_based import EvoForestTSWM


def _fixture_X():
    rng = np.random.RandomState(42)
    return np.sin(np.linspace(0, 6, 80))[None, None, :] * np.array([1, 2, 3])[
        :, None, None
    ] + 0.1 * rng.normal(size=(3, 1, 80))


# frozen aggregates of the discovered champion on _fixture_X() (torch-free)
_EXPECT = {
    "full": dict(
        shape=(3, 519),
        agg=[1060.464338, 0.681095, 1.925746, 0.030663, 0.37177, -8.210593, 13.586488],
    ),
    "pruned": dict(
        shape=(3, 211),
        agg=[509.801995, 0.805374, 1.865634, -0.468255, 0.028522, -5.012907, 13.586488],
    ),
}


def _agg(out):
    return [
        float(out.sum()),
        float(out.mean()),
        float(out.std()),
        float(out[0, 0]),
        float(out[-1, -1]),
        float(out.min()),
        float(out.max()),
    ]


@pytest.mark.parametrize("pooling", ["full", "pruned"])
def test_evoforest_tswm_shape_and_values(pooling):
    """Output shape and a frozen numeric regression lock the discovered encoder."""
    out = EvoForestTSWM(pooling=pooling).fit_transform(_fixture_X())
    exp = _EXPECT[pooling]
    assert out.shape == exp["shape"]
    assert np.isfinite(out).all()
    np.testing.assert_allclose(_agg(out), exp["agg"], rtol=1e-4, atol=1e-4)


def test_evoforest_tswm_multivariate_width_invariant():
    """Output width is independent of the channel count."""
    rng = np.random.RandomState(0)
    f1 = EvoForestTSWM().fit_transform(rng.normal(size=(4, 1, 70)))
    f3 = EvoForestTSWM().fit_transform(rng.normal(size=(4, 3, 70)))
    assert f1.shape[1] == f3.shape[1] == 519


def test_evoforest_tswm_bad_pooling():
    """An invalid pooling argument raises ValueError."""
    with pytest.raises(ValueError, match="pooling"):
        EvoForestTSWM(pooling="nope").fit_transform(_fixture_X())
