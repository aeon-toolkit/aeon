"""Wrapper for imblearn minority class rebalancer SMOTE."""

from imblearn.over_sampling import ADASYN as adasyn
import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer

__maintainer__ = ["TonyBagnall, Chris Qiu"]
__all__ = ["ADASYN"]


class ADASYN(BaseCollectionTransformer):
    """Wrapper for ADASYN transform."""

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "requires_y": True,
    }

    def __init__(self, sampling_strategy="auto", random_state=None, k_neighbors=5):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        super().__init__()

    def _transform(self, X, y=None):
        return self

    def _fit_transform(self, X, y=None):
        self.smote_ = adasyn(sampling_strategy=self.sampling_strategy,
                             random_state=self.random_state, n_neighbors=self.k_neighbors)
        res_X, res_y = self.smote_.fit_resample(np.squeeze(X), y)
        return res_X, res_y


if __name__ == "__main__":
    # Example usage
    import numpy as np
    n_samples = 100  # Total number of labels
    imbalance_ratio = 0.9  # Proportion of majority class

    X = np.random.rand(n_samples, 1, 10)
    y = np.random.choice([0, 1], size=n_samples, p=[imbalance_ratio, 1 - imbalance_ratio])

    _, count = np.unique(y, return_counts=True)
    print(count)
    transformer = ADASYN()
    res_X, res_y = transformer.fit_transform(X, y)
    print(res_X.shape, res_y.shape)
    # Expected output: (200, 3, 10) (200,)
    _, res_count = np.unique(res_y, return_counts=True)
    print(res_count)