"""Channel selection test code."""

from aeon.testing.utils.data_gen import make_example_3d_numpy
from aeon.transformations.collection.channel_selection import ElbowClassPairwise


def test_channel_selection():
    """Test channel selection on random nested data frame."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=4, n_timepoints=20)

    ecp = ElbowClassPairwise()

    ecp.fit(X, y)
    Xt = ecp.transform(X, y)

    # test shape of transformed data should be
    # (n_samples, n_channels_selected, n_timepoints)
    assert Xt.shape == (X.shape[0], len(ecp.channels_selected_idx), X.shape[2])
