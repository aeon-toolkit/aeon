# Using aeon with scikit-learn style workflows

This example shows how aeon can be used in a way that feels familiar to
users of scikit-learn.

We demonstrate a simple time-series classification example using
`KNeighborsTimeSeriesClassifier`.

## Example: Time Series Classification

```python
import numpy as np
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

# Training data: (n_cases, n_channels, n_timepoints)
X_train = np.array([
    [[1, 2, 3, 4, 5, 6]],
    [[1, 2, 2, 3, 3, 4]],
    [[6, 5, 4, 3, 2, 1]],
])

y_train = np.array(["low", "low", "high"])

clf = KNeighborsTimeSeriesClassifier(distance="dtw")
clf.fit(X_train, y_train)

# Test data
X_test = np.array([
    [[2, 2, 3, 3, 4, 4]],
    [[5, 4, 3, 2, 1, 1]],
])

predictions = clf.predict(X_test)
print(predictions)
