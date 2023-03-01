# -*- coding: utf-8 -*-
"""Channel selection test code."""
from sktime.transformations.panel.channel_selection import ElbowClassPairwise
from sktime.utils._testing.panel import make_classification_problem


def test_cs_basic_motions():
    """Test channel selection on random nested data frame."""
    X, y = make_classification_problem(n_instances=10, n_timepoints=10, n_columns=3)

    ecp = ElbowClassPairwise()

    ecp.fit(X, y)

    ecp.transform(X, y)

    # test shape pf transformed data should be (n_samples, n_channels_selected)
    assert ecp.transform(X, y).shape == (X.shape[0], len(ecp.channels_selected_idx))

    # test shape of transformed data should be (n_samples, n_channels_selected)

    X_test, y_test = make_classification_problem(
        n_instances=10, n_timepoints=10, n_columns=3
    )

    assert ecp.transform(X_test).shape == (
        X_test.shape[0],
        len(ecp.channels_selected_idx),
    )
