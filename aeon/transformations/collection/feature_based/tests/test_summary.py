"""Test summary features transformer."""

import numpy as np
import pytest

from aeon.transformations.collection.feature_based import SevenNumberSummary


@pytest.mark.parametrize("summary_stats", ["default", "quantiles", "bowley", "tukey"])
def test_summary_features(summary_stats):
    """Test different summary_stats options."""
    sns = SevenNumberSummary()
    t = sns.fit_transform(np.ones((10, 2, 5)))
    assert t.shape == (10, 14)


def test_summary_features_invalid():
    """Test invalid summary_stats option."""
    with pytest.raises(ValueError, match="Summary function input invalid"):
        sns = SevenNumberSummary(summary_stats="invalid")
        sns.fit_transform(np.ones((10, 2, 5)))
