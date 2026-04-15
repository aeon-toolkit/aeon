<a href="https://www.aeon-toolkit.org/">
  <img src="https://raw.githubusercontent.com/aeon-toolkit/aeon/main/docs/images/logo/aeon-logo-blue-compact.png" alt="aeon logo" width="400">
</a>

**Time series machine learning, built by the researchers behind the algorithms.**

`aeon` is a scikit-learn compatible Python library for learning from time series.
It covers classification, regression, clustering, forecasting, anomaly detection,
segmentation, similarity search, and transformation — with implementations
contributed and maintained by the researchers who designed many of the methods,
including state-of-the-art deep learning models for forecasting, classification,
regression and clustering.

[Documentation](https://www.aeon-toolkit.org/) ·
[Examples](https://www.aeon-toolkit.org/en/stable/examples.html) ·
[API reference](https://www.aeon-toolkit.org/en/stable/api_reference.html) ·
[Getting started](https://www.aeon-toolkit.org/en/stable/getting_started.html) ·
[Discussions](https://github.com/aeon-toolkit/aeon/discussions) ·
[Discord](https://discord.gg/52F5RAGD)

[![PyPI version](https://img.shields.io/pypi/v/aeon)](https://pypi.org/project/aeon/)
[![Python versions](https://img.shields.io/pypi/pyversions/aeon)](https://pypi.org/project/aeon/)
[![License](https://img.shields.io/github/license/aeon-toolkit/aeon)](https://github.com/aeon-toolkit/aeon/blob/main/LICENSE)
[![Docs](https://readthedocs.org/projects/aeon/badge/?version=latest)](https://www.aeon-toolkit.org/)
[![Tests](https://github.com/aeon-toolkit/aeon/actions/workflows/tests.yml/badge.svg)](https://github.com/aeon-toolkit/aeon/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/codecov/c/github/aeon-toolkit/aeon)](https://codecov.io/gh/aeon-toolkit/aeon)
[![Discord](https://img.shields.io/discord/930275473265115166?label=discord)](https://discord.gg/52F5RAGD)

> 📄 Published in the **Journal of Machine Learning Research** (2024) —
> [aeon: a Python Toolkit for Learning from Time Series](http://jmlr.org/papers/v25/23-1444.html)

## From paper to `pip install`

`aeon` is developed in close contact with the time series research community.
Many of its algorithms are contributed or maintained by their original authors,
and the same team behind `aeon` runs the benchmarks that the field uses to
evaluate new methods. That means:

- **Faithful implementations.** Algorithms reflect what the papers actually describe.
- **State of the art, sooner.** New methods often land in `aeon` alongside publication.
- **Evidence-based defaults.** What's included — and what's recommended — is grounded in published comparative studies.

A selection of algorithms available in `aeon`  and comparative studies based on aeon written by aeon core developers or contributors:

| Method                           | Reference                                                                                      | Task           |
|----------------------------------|------------------------------------------------------------------------------------------------|----------------|
| **LITETime/InceptionTime**       | Ismail-Fawaz et al., [2020](https://link.springer.com/article/10.1007/s10618-020-00710-y)/2023 | Classification |
| **Hydra-MultiRocket**            | [Dempster et al., 2023](https://link.springer.com/article/10.1007/s10618-023-00939-3)          | Classification |
| **SETAR-Tree**                   | [Godahewa et al. 2023](https://link.springer.com/article/10.1007/s10994-023-06316-x)           | Forecasting    |
| **KASBA**                        | [Holder et al. 2026](https://link.springer.com/article/10.1007/s10618-026-01189-9)             | Clustering     |
| **CLASP**                        | [Ermshaus et al. 2023](https://link.springer.com/article/10.1007/s10618-023-00923-x)           | Segmentation   |
| Ordinal classification methods   | [Guijo-Rubio et al. 2025](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10769513)       | Classification |
| Multivariate classification      | [Middlehurst et al.2026](https://arxiv.org/abs/2603.20352)                                     | Classification |
| Classification (the "Bake off")  | [Bagnall et al., 2017](https://link.springer.com/article/10.1007/S10618-016-0483-9)            | Benchmarking   |
| Bake off redux                   | [Middlehurst et al., 2025](https://link.springer.com/article/10.1007/S10618-016-0483-9)        | Benchmarking   |
| Deep learning for classification | [Ismail-Fawaz et al.](https://link.springer.com/article/10.1007/s10618-019-00619-1)            | Benchmarking   |

<h4>Supported by</h4>
<p>
  <a href="https://www.inria.fr/">
    <img src="https://raw.githubusercontent.com/aeon-toolkit/aeon/main/docs/images/logos/inria.png" alt="INRIA" height="50">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://www.ukri.org/councils/epsrc/">
    <img src="https://raw.githubusercontent.com/aeon-toolkit/aeon/main/docs/images/logos/epsrc.png" alt="EPSRC" height="50">
  </a>
</p>

See the [API reference](https://www.aeon-toolkit.org/en/stable/api_reference.html)
for the full list across all tasks.

⭐ **Star the repo** to follow new releases — `aeon` ships frequently, and starring is the easiest way to know when new algorithms land.

## Deep learning for time series

`aeon` provides Keras/TensorFlow implementations of leading deep learning
architectures for time series, with a consistent scikit-learn compatible API
and many models contributed by their original authors:

- **Classification:** InceptionTime, H-InceptionTime, LITE, LITETime, ResNet, FCN, MLP, CNN, Disjoint-CNN, and more
- **Regression:** the same backbone architectures, adapted for continuous targets
- **Clustering:** deep learning based clustering via learned representations

A minimal example:

```python
from aeon.datasets import load_unit_test
from aeon.classification.deep_learning import InceptionTimeClassifier

X_train, y_train = load_unit_test(split="train")
X_test, y_test = load_unit_test(split="test")

clf = InceptionTimeClassifier(n_epochs=10)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
```

See the [examples gallery](https://www.aeon-toolkit.org/en/stable/examples.html)
for GPU usage, custom architectures, and benchmarking against classical methods.

## Installation

`aeon` requires Python 3.10 or newer.

Install the latest release from PyPI:

```
pip install aeon
```

To install with all optional dependencies (including deep learning):

```
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
# 1.0
```

## Tasks supported

Ten task areas, one consistent API:

| Task | What it does | |
| --- | --- | --- |
| **Classification** | Predict labels for time series | [docs →](https://www.aeon-toolkit.org/en/stable/api_reference/classification.html) |
| **Regression** | Predict continuous values from time series | [docs →](https://www.aeon-toolkit.org/en/stable/api_reference/regression.html) |
| **Clustering** | Group similar series without labels | [docs →](https://www.aeon-toolkit.org/en/stable/api_reference/clustering.html) |
| **Forecasting** | Predict future values | [docs →](https://www.aeon-toolkit.org/en/stable/api_reference/forecasting.html) |
| **Anomaly detection** | Find unusual points or subsequences | [docs →](https://www.aeon-toolkit.org/en/stable/api_reference/anomaly_detection.html) |
| **Segmentation** | Split a series into homogeneous regions | [docs →](https://www.aeon-toolkit.org/en/stable/api_reference/segmentation.html) |
| **Similarity search** | Find similar subsequences in long series | [docs →](https://www.aeon-toolkit.org/en/stable/api_reference/similarity_search.html) |
| **Transformations** | Feature extraction and preprocessing | [docs →](https://www.aeon-toolkit.org/en/stable/api_reference/transformations.html) |
| **Distances & kernels** | Time series similarity measures | [docs →](https://www.aeon-toolkit.org/en/stable/api_reference/distances.html) |
| **Benchmarking** | Reproducible experimental evaluation | [docs →](https://www.aeon-toolkit.org/en/stable/api_reference/benchmarking.html) |

## Getting started examples

### Classification and regression

Time series classification predicts class labels for unseen series using a model fitted on a collection of labelled time series. Regression follows the same pattern, but predicts continuous values instead of class labels.

```python
import numpy as np
from aeon.classification.convolution_based import MultiRocketHydraClassifier

X = np.array([
    [[1, 2, 3, 4, 5, 5]],
    [[1, 2, 3, 4, 4, 2]],
    [[8, 7, 6, 5, 4, 4]],
])
y = np.array(["low", "low", "high"])

clf = MultiRocketHydraClassifier(n_kernels=1000)
clf.fit(X, y)

X_test = np.array([
    [[2, 2, 2, 2, 2, 2]],
    [[5, 5, 5, 5, 5, 5]],
    [[6, 6, 6, 6, 6, 6]],
])

y_pred = clf.predict(X_test)
print(y_pred)
# ['low' 'low' 'high']
```

### Clustering

Time series clustering groups similar time series together from an unlabelled collection.

```python
import numpy as np
from aeon.clustering import KASBA

X = np.array([
    [[1, 2, 3, 4, 5, 5]],
    [[1, 2, 3, 4, 4, 2]],
    [[8, 7, 6, 5, 4, 4]],
])

clu = KASBA(n_clusters=2)
clu.fit(X)

print(clu.labels_)
```

### Forecasting

`aeon` provides a wide range of forecasting algorithms, including classic
statistical models and modern deep learning approaches.

<!-- TODO: replace this example with a more aeon-distinctive forecaster -->

```python
from aeon.datasets import load_airline
from aeon.forecasting.machine_learning import SETARForest

y = load_airline()

forecaster = SETARForest(n_estimators=10)
pred = forecaster.forecast(y)

print(pred)
```

For more examples across tasks, visit the
[examples gallery](https://www.aeon-toolkit.org/en/stable/examples.html).

## Support aeon

There are several simple ways to support the project:

- ⭐ **Star this repository** to follow new releases
- 👀 **Watch releases** for notifications about new features
- 📝 **Cite `aeon`** in academic work
- 🐛 **Report bugs and request features** through GitHub Issues and Discussions
- 🛠️ **Contribute** code, tests, documentation, or examples

## Where to ask questions

- **Bug reports and feature requests** → [GitHub Issues](https://github.com/aeon-toolkit/aeon/issues)
- **Usage questions and general discussion** → [GitHub Discussions](https://github.com/aeon-toolkit/aeon/discussions) or [Discord](https://discord.gg/52F5RAGD)

For project or collaboration enquiries, contact **[contact@aeon-toolkit.org](mailto:contact@aeon-toolkit.org)**.

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
  author  = {Matthew Middlehurst and Ali Ismail-Fawaz and Antoine Guillaume and Christopher Holder and David Guijo-Rubio and Guzal Bulatova and Leonidas Tsaprounis and Lukasz Mentel and Martin Walter and Patrick Sch{\"a}fer and Anthony Bagnall},
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

## Project history

`aeon` was forked from `sktime` `v0.16.0` in 2022 by an initial group of core developers, and has since been substantially rewritten and extended. You can read more about the project's history, values, and governance on the [About Us page](https://www.aeon-toolkit.org/en/stable/about.html).

## Project status

`aeon` is under active development. The core package is stable and widely used. The following modules are currently considered in development, and the deprecation policy does not necessarily apply (although we only rarely make non-compatible changes): `anomaly_detection`, `forecasting`, `segmentation`, `similarity_search`, `visualisation`, `transformations.collection.self_supervised`, `transformations.collection.imbalance`.

Please check the documentation for task-specific capabilities, limitations, and current status.
