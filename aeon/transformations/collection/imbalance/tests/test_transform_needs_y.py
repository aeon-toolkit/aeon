"""Every sampler asks for the labels rather than failing on them."""

import numpy as np
import pytest

from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.imbalance import (
    ADASYN,
    ESMOTE,
    OHIT,
    SMOTE,
    RandomOverSampler,
)

SAMPLERS = [ADASYN, ESMOTE, OHIT, SMOTE, RandomOverSampler]


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_transform_without_y_is_refused(sampler):
    """A sampler resamples X and y together, so transform cannot run on X alone.

    The base signature makes y optional and every sampler read it straight into
    `y.copy()`, so the call raised AttributeError on None rather than naming the
    argument it wanted.
    """
    X, y = make_example_3d_numpy(n_cases=20, n_channels=1, n_labels=2, random_state=0)
    fitted = sampler().fit(X, y)
    with pytest.raises(ValueError, match="resamples X and y together"):
        fitted.transform(X)


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_transform_with_y_still_resamples(sampler):
    """The guard leaves the supported call alone."""
    X, _ = make_example_3d_numpy(n_cases=20, n_channels=1, random_state=0)
    y = np.array([0] * 14 + [1] * 6)
    fitted = sampler().fit(X, y)
    Xt, yt = fitted.transform(X, y)
    assert len(Xt) == len(yt)
    assert len(yt) >= len(y)
