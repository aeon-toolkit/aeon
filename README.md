<div align="center">
  <img src="https://raw.githubusercontent.com/aeon-toolkit/aeon/main/docs/images/aeon_logo.png" width="220" alt="aeon logo" />

# aeon

**A Python toolkit for time series machine learning**

Unified, efficient, and extensible tools for learning from time series data in Python.
`aeon` provides a consistent, scikit-learn compatible interface for tasks including
classification, regression, clustering, forecasting, anomaly detection, segmentation,
similarity search, transformation, and benchmarking.

[Documentation](https://www.aeon-toolkit.org/) |
[Examples](https://www.aeon-toolkit.org/en/stable/examples.html) |
[API reference](https://www.aeon-toolkit.org/en/stable/api_reference.html) |
[Getting started](https://www.aeon-toolkit.org/en/stable/getting_started.html) |
[GitHub Discussions](https://github.com/aeon-toolkit/aeon/discussions) |
[Discord](https://discord.gg/52F5RAGD)

[![PyPI version](https://img.shields.io/pypi/v/aeon)](https://pypi.org/project/aeon/)
[![Python versions](https://img.shields.io/pypi/pyversions/aeon)](https://pypi.org/project/aeon/)
[![License](https://img.shields.io/github/license/aeon-toolkit/aeon)](https://github.com/aeon-toolkit/aeon/blob/main/LICENSE)
[![Docs](https://readthedocs.org/projects/aeon/badge/?version=latest)](https://www.aeon-toolkit.org/)
[![Tests](https://github.com/aeon-toolkit/aeon/actions/workflows/tests.yml/badge.svg)](https://github.com/aeon-toolkit/aeon/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/codecov/c/github/aeon-toolkit/aeon)](https://codecov.io/gh/aeon-toolkit/aeon)
[![Discord](https://img.shields.io/discord/930275473265115166?label=discord)](https://discord.gg/52F5RAGD)

</div>

## Why aeon?

- **Broad task coverage** across the main areas of time series machine learning in one library
- **Consistent APIs** inspired by scikit-learn design principles
- **Efficient implementations** with a focus on practical performance and reproducible evaluation
- **Research-friendly design** for experimentation, extension, and benchmarking
- **Open development** with documentation, examples, releases, and community support

## Installation

`aeon` requires Python 3.10 or newer.

Install the latest release from PyPI:

```bash
pip install aeon
```

To install with all optional dependencies:

```bash
pip install aeon[all_extras]
```

For development installs and platform-specific notes, see the
[installation guide](https://www.aeon-toolkit.org/en/stable/installation.html).

## Quick start

The fastest way to try `aeon` is to fit a classifier on a built-in dataset.

```python
from aeon.datasets import load_unit_test
from aeon.classification.convolution_based import RocketClassifier

X_train, y_train = load_unit_test(split="train")
X_test, y_test = load_unit_test(split="test")

clf = RocketClassifier()
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
```

If `aeon` is useful in your work, please **star the repository**. It helps more people discover the project, helps signal community interest, and supports the long-term visibility of the toolkit.

## What can you do with aeon?

`aeon` supports a wide range of time series learning tasks and utilities, including:

- classification
- regression
- clustering
- forecasting
- anomaly detection
- segmentation
- similarity search
- transformations and preprocessing
- distances, kernels, and similarity measures
- benchmarking and performance evaluation

See the [documentation](https://www.aeon-toolkit.org/) for tutorials, user guides, and full API coverage.

## Getting started examples

### Classification and regression

Time series classification predicts class labels for unseen series using a model fitted on a collection of labelled time series. Regression follows the same pattern, but predicts continuous values instead of class labels.

```python
import numpy as np
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

X = np.array([
    [[1, 2, 3, 4, 5, 5]],
    [[1, 2, 3, 4, 4, 2]],
    [[8, 7, 6, 5, 4, 4]],
])
y = np.array(["low", "low", "high"])

clf = KNeighborsTimeSeriesClassifier(distance="dtw")
clf.fit(X, y)

X_test = np.array([
    [[2, 2, 2, 2, 2, 2]],
    [[5, 5, 5, 5, 5, 5]],
    [[6, 6, 6, 6, 6, 6]],
])

y_pred = clf.predict(X_test)
print(y_pred)
```

### Clustering

Time series clustering groups similar time series together from an unlabelled collection.

```python
import numpy as np
from aeon.clustering import TimeSeriesKMeans

X = np.array([
    [[1, 2, 3, 4, 5, 5]],
    [[1, 2, 3, 4, 4, 2]],
    [[8, 7, 6, 5, 4, 4]],
])

clu = TimeSeriesKMeans(distance="dtw", n_clusters=2)
clu.fit(X)

print(clu.labels_)
```

For more examples across tasks, visit the
[examples gallery](https://www.aeon-toolkit.org/en/stable/examples.html).

## Support aeon

There are several simple ways to support the project:

- **Star this repository** if you use or follow `aeon`
- **Watch releases** to stay informed about new features and improvements
- **Cite `aeon`** in academic work
- **Report bugs and request features** through GitHub Issues and Discussions
- **Contribute code, tests, documentation, or examples**

## Where to ask questions

| Type | Platforms |
| --- | --- |
| Bug reports | [GitHub Issues](https://github.com/aeon-toolkit/aeon/issues) |
| Feature requests and ideas | [GitHub Issues](https://github.com/aeon-toolkit/aeon/issues) and [Discord](https://discord.gg/52F5RAGD) |
| Usage questions | [GitHub Discussions](https://github.com/aeon-toolkit/aeon/discussions) and [Discord](https://discord.gg/52F5RAGD) |
| General discussion | [GitHub Discussions](https://github.com/aeon-toolkit/aeon/discussions) and [Discord](https://discord.gg/52F5RAGD) |
| Contribution and development | [GitHub Discussions](https://github.com/aeon-toolkit/aeon/discussions) and [Discord](https://discord.gg/52F5RAGD) |

For project or collaboration enquiries, contact **contact@aeon-toolkit.org**.

## Contributing to aeon

If you are interested in contributing, please read the
[contributing guide](https://github.com/aeon-toolkit/aeon/blob/main/CONTRIBUTING.md)
before opening a pull request or taking ownership of an issue.

Useful links:

- [Contributing guide](https://github.com/aeon-toolkit/aeon/blob/main/CONTRIBUTING.md)
- [Code of conduct](https://github.com/aeon-toolkit/aeon/blob/main/CODE_OF_CONDUCT.md)
- [Governance](https://github.com/aeon-toolkit/aeon/blob/main/GOVERNANCE.md)
- [Project website](https://www.aeon-toolkit.org/)

The `aeon` developers are volunteers, so please be patient with issue triage and pull request review.

## Citation

If you use `aeon` in academic work, please cite the project:

```bibtex
@article{aeon24jmlr,
  author  = {Matthew Middlehurst and Ali Ismail-Fawaz and Antoine Guillaume and Christopher Holder and David Guijo-Rubio and Guzal Bulatova and Leonidas Tsaprounis and Lukasz Mentel and Martin Walter and Patrick Sch{"a}fer and Anthony Bagnall},
  title   = {aeon: a Python Toolkit for Learning from Time Series},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {289},
  pages   = {1--10},
  url     = {http://jmlr.org/papers/v25/23-1444.html}
}
```

You can also use the repository's [CITATION.cff](https://github.com/aeon-toolkit/aeon/blob/main/CITATION.cff).

If you let us know about your paper using `aeon`, we will happily list it on the project website.

## Further information

`aeon` was forked from `sktime` `v0.16.0` in 2022 by an initial group of core developers. You can read more about the project's history, values, and governance on the [About Us page](https://www.aeon-toolkit.org/en/stable/about.html).

## Project status

`aeon` is under active development. The core package is stable and widely used, but some modules and recently added functionality remain experimental and may change as the library evolves.

The following modules are currently considered experimental, and the deprecation
policy does not necessarily apply (although we only rarely make rapid):

- `anomaly_detection`
- `forecasting`
- `segmentation`
- `similarity_search`
- `visualisation`
- `transformations.collection.self_supervised`
- `transformations.collection.imbalance`

Please check the documentation for task-specific capabilities, limitations, and current status.
