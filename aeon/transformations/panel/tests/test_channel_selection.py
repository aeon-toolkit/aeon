# -*- coding: utf-8 -*-
"""Channel selection test code."""
from aeon.transformations.panel.channel_selection import ElbowClassPairwise
from aeon.utils._testing.panel import make_classification_problem


def test_cs_basic_motions():
    """Test channel selection on random nested data frame."""
    X_train, y_train = make_classification_problem(
        n_instances=10, n_timepoints=10, n_columns=3
    )
    X_test, y_test = make_classification_problem(
        n_instances=10, n_timepoints=10, n_columns=3
    )

    ecp = ElbowClassPairwise()

    # shape of transformed data should be the same except for possibly dimensions
    assert ecp.fit_transform(X_train, y_train).shape == (
        X_train.shape[0],
        len(ecp.channels_selected_idx),
        X_train.shape[2],
    )
    assert ecp.transform(X_test).shape == (
        X_test.shape[0],
        len(ecp.channels_selected_idx),
        X_test.shape[2],
    )
