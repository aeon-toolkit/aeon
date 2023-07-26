<p align="center">
    <a href="https://aeon-toolkit.org"><img src="https://github.com/aeon-toolkit/aeon/blob/main/docs/images/logo/aeon-logo-blue-compact.png" width="50%" alt="aeon logo" /></a>
</p>

# ⌛ Welcome to aeon

`aeon` is an open source toolkit for learning from time series. It is compatible with
[scikit-learn](https://scikit-learn.org) and provides access to the very latest
algorithms for time series  machine learning, in addition to a range of classical
techniques for learning tasks such as forecasting and classification.

We strive to provide a broad library of time series algorithms including the
latest advances, offer efficient implementations using numba, and interfaces with other
time series packages to provide a single framework for algorithm comparison.

The latest `aeon` release is `v0.4.0`. You can view the full changelog
[here](https://www.aeon-toolkit.org/en/latest/changelog.html).

```diff
- The deprecation policy is currently suspended, be careful with the version bounds used when including aeon as a dependency.
- The policy will return at a future point, but in the mean time the suspension allows us to quickly develop and maintain the package in the forking transition period.
```

Our webpage and documentation is available at https://aeon-toolkit.org.

| Overview      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CI/CD**     | [![github-actions-release](https://img.shields.io/github/actions/workflow/status/aeon-toolkit/aeon/release.yml?logo=github&label=build%20%28release%29)](https://github.com/aeon-toolkit/aeon/actions/workflows/release.yml) [![github-actions-main](https://img.shields.io/github/actions/workflow/status/aeon-toolkit/aeon/pr_tests.yml?logo=github&branch=main&label=build%20%28main%29)](https://github.com/aeon-toolkit/aeon/actions/workflows/pr_tests.yml) [![github-actions-nightly](https://img.shields.io/github/actions/workflow/status/aeon-toolkit/aeon/periodic_tests.yml?logo=github&label=build%20%28nightly%29)](https://github.com/aeon-toolkit/aeon/actions/workflows/periodic_tests.yml) [![docs-main](https://img.shields.io/readthedocs/aeon-toolkit/stable?logo=readthedocs&label=docs%20%28stable%29)](https://www.aeon-toolkit.org/en/stable/) [![docs-main](https://img.shields.io/readthedocs/aeon-toolkit/latest?logo=readthedocs&label=docs%20%28latest%29)](https://www.aeon-toolkit.org/en/latest/) [![!codecov](https://img.shields.io/codecov/c/github/aeon-toolkit/aeon?label=codecov&logo=codecov)](https://codecov.io/gh/aeon-toolkit/aeon) |
| **Code**      | [![!pypi](https://img.shields.io/pypi/v/aeon?logo=pypi&color=blue)](https://pypi.org/project/aeon/) [![!python-versions](https://img.shields.io/pypi/pyversions/aeon?logo=python)](https://www.python.org/) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![license](https://img.shields.io/badge/license-BSD%203--Clause-green?logo=style)](https://github.com/aeon-toolkit/aeon/blob/main/LICENSE) [![binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aeon-toolkit/aeon/main?filepath=examples)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **Community** | [![!slack](https://img.shields.io/static/v1?logo=slack&label=Slack&message=chat&color=lightgreen)](https://join.slack.com/t/aeon-toolkit/shared_invite/zt-1plkevy4x-vAg1dAUXcuoR38FjY9nxzg) [![!linkedin](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/aeon-toolkit/) [![!twitter](https://img.shields.io/static/v1?logo=twitter&label=Twitter&message=news&color=lightblue)](https://twitter.com/aeon_toolkit)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |

## ⚙️ Installation

`aeon` requires a Python version of 3.8 or greater. Our full installation guide is
available in our [documentation](https://www.aeon-toolkit.org/en/latest/installation.html).

The easiest way to install `aeon` is via pip:

```bash
pip install aeon
```

Some estimators require additional packages to be installed. If you want to install
the full package with all optional dependencies, you can use:

```bash
pip install aeon[all_extras]
```
## ⏲️ Getting started

The best place to started for all `aeon` packages is our [getting started guide](https://www.aeon-toolkit.org/en/latest/getting_started.html).

Below we provide a quick example of how to use `aeon` for forecasting and
classification.

### Forecasting

```python
import pandas as pd
from aeon.forecasting.trend import TrendForecaster

y = pd.Series([20.0, 40.0, 60.0, 80.0, 100.0])
>>> 0     20.0
>>> 1     40.0
>>> 2     60.0
>>> 3     80.0
>>> 4    100.0
>>> dtype: float64

forecaster = TrendForecaster()
forecaster.fit(y)  # fit the forecaster
>>> TrendForecaster()

pred = forecaster.predict(fh=[1, 2, 3])  # forecast the next 3 values
>>> 5    120.0
>>> 6    140.0
>>> 7    160.0
>>> dtype: float64
```

### Classification

```python
import numpy as np
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

X = [[[1, 2, 3, 4, 5, 5]],  # 3D array example (univariate)
     [[1, 2, 3, 4, 4, 2]],  # Three samples, one channel, six series length,
     [[8, 7, 6, 5, 4, 4]]]
y = ['low', 'low', 'high']  # class labels for each sample
X = np.array(X)
y = np.array(y)

clf = KNeighborsTimeSeriesClassifier(distance="dtw")
clf.fit(X, y)  # fit the classifier on train data
>>> KNeighborsTimeSeriesClassifier()

X_test = np.array(
    [[[2, 2, 2, 2, 2, 2]], [[5, 5, 5, 5, 5, 5]], [[6, 6, 6, 6, 6, 6]]]
)
y_pred = clf.predict(X_test)  # make class predictions on new data
>>> ['low' 'high' 'high']
```

## 💬 Where to ask questions

| Type                                | Platforms                        |
|-------------------------------------|----------------------------------|
| 🐛 **Bug Reports**                  | [GitHub Issue Tracker]           |
| ✨ **Feature Requests & Ideas**      | [GitHub Issue Tracker] & [Slack] |
| 💻 **Usage Questions**              | [GitHub Discussions] & [Slack]   |
| 💬 **General Discussion**           | [GitHub Discussions] & [Slack]   |
| 🏭 **Contribution & Development**   | [Slack]                          |

[GitHub Issue Tracker]: https://github.com/aeon-toolkit/aeon/issues
[GitHub Discussions]: https://github.com/aeon-toolkit/aeon/discussions
[Slack]: https://join.slack.com/t/aeon-toolkit/shared_invite/zt-1plkevy4x-vAg1dAUXcuoR38FjY9nxzg
