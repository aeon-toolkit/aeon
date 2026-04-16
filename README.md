<a href="https://www.aeon-toolkit.org/">
  <img src="https://raw.githubusercontent.com/aeon-toolkit/aeon/main/docs/images/logo/aeon-logo-blue-compact.png" alt="aeon logo" width="400">
</a>

**Time series machine learning, built by the researchers behind the algorithms.**

`aeon` is a scikit-learn compatible Python library for learning from time series.
It covers classification, regression, clustering, forecasting, anomaly detection, distance functions,
segmentation, similarity search, transformation and benchmarking.

Many implementations in aeon are contributed and maintained by the researchers who developed the original methods. This includes a range of modern and state-of-the-art deep learning models for forecasting, classification, regression, and clustering.

[Documentation](https://www.aeon-toolkit.org/) ·
[Examples](https://www.aeon-toolkit.org/en/stable/examples.html) ·
[API reference](https://www.aeon-toolkit.org/en/stable/api_reference.html) ·
[Getting started](https://www.aeon-toolkit.org/en/stable/getting_started.html) ·
[Discussions](https://github.com/aeon-toolkit/aeon/discussions) ·
[Discord](https://discord.gg/D6rzqHGKRJ)

> 📄 Published in the **Journal of Machine Learning Research** (2024) —
> [aeon: a Python Toolkit for Learning from Time Series](http://jmlr.org/papers/v25/23-1444.html)

| Overview        |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|-----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CI/CD**       | [![github-actions-release](https://img.shields.io/github/actions/workflow/status/aeon-toolkit/aeon/release.yml?logo=github&label=build%20%28release%29)](https://github.com/aeon-toolkit/aeon/actions/workflows/release.yml) [![github-actions-main](https://img.shields.io/github/actions/workflow/status/aeon-toolkit/aeon/pr_pytest.yml?logo=github&branch=main&label=build%20%28main%29)](https://github.com/aeon-toolkit/aeon/actions/workflows/pr_pytest.yml) [![github-actions-nightly](https://img.shields.io/github/actions/workflow/status/aeon-toolkit/aeon/periodic_tests.yml?logo=github&label=build%20%28nightly%29)](https://github.com/aeon-toolkit/aeon/actions/workflows/periodic_tests.yml) [![docs-main](https://img.shields.io/readthedocs/aeon-toolkit/stable?logo=readthedocs&label=docs%20%28stable%29)](https://www.aeon-toolkit.org/en/stable/) [![docs-main](https://img.shields.io/readthedocs/aeon-toolkit/latest?logo=readthedocs&label=docs%20%28latest%29)](https://www.aeon-toolkit.org/en/latest/) [![codecov](https://codecov.io/gh/aeon-toolkit/aeon/graph/badge.svg?token=I2eve2HzSF)](https://codecov.io/gh/aeon-toolkit/aeon) [![openssf-scorecard](https://api.scorecard.dev/projects/github.com/aeon-toolkit/aeon/badge)](https://scorecard.dev/viewer/?uri=github.com/aeon-toolkit/aeon) |
| **Code**        | [![!pypi](https://img.shields.io/pypi/v/aeon?logo=pypi&color=blue)](https://pypi.org/project/aeon/) [![!conda](https://img.shields.io/conda/vn/conda-forge/aeon?logo=anaconda&color=blue)](https://anaconda.org/conda-forge/aeon) [![!python-versions](https://img.shields.io/pypi/pyversions/aeon?logo=python)](https://www.python.org/) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![license](https://img.shields.io/badge/license-BSD%203--Clause-green?logo=style)](https://github.com/aeon-toolkit/aeon/blob/main/LICENSE) [![binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aeon-toolkit/aeon/main?filepath=examples)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Community**   | [![!discord](https://img.shields.io/static/v1?logo=discord&label=discord&message=chat&color=lightgreen)](https://discord.gg/D6rzqHGKRJ) [![!linkedin](https://img.shields.io/static/v1?logo=data:image/svg%2bxml;base64,PHN2ZyByb2xlPSJpbWciIGZpbGw9IiNmZmZmZmYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48dGl0bGU+TGlua2VkSW48L3RpdGxlPjxwYXRoIGQ9Ik0yMC40NDcgMjAuNDUyaC0zLjU1NHYtNS41NjljMC0xLjMyOC0uMDI3LTMuMDM3LTEuODUyLTMuMDM3LTEuODUzIDAtMi4xMzYgMS40NDUtMi4xMzYgMi45Mzl2NS42NjdIOS4zNTFWOWgzLjQxNHYxLjU2MWguMDQ2Yy40NzctLjkgMS42MzctMS44NSAzLjM3LTEuODUgMy42MDEgMCA0LjI2NyAyLjM3IDQuMjY3IDUuNDU1djYuMjg2ek01LjMzNyA3LjQzM2MtMS4xNDQgMC0yLjA2My0uOTI2LTIuMDYzLTIuMDY1IDAtMS4xMzguOTItMi4wNjMgMi4wNjMtMi4wNjMgMS4xNCAwIDIuMDY0LjkyNSAyLjA2NCAyLjA2MyAwIDEuMTM5LS45MjUgMi4wNjUtMi4wNjQgMi4wNjV6bTEuNzgyIDEzLjAxOUgzLjU1NVY5aDMuNTY0djExLjQ1MnpNMjIuMjI1IDBIMS43NzFDLjc5MiAwIDAgLjc3NCAwIDEuNzI5djIwLjU0MkMwIDIzLjIyNy43OTIgMjQgMS43NzEgMjRoMjAuNDUxQzIzLjIgMjQgMjQgMjMuMjI3IDI0IDIyLjI3MVYxLjcyOUMyNCAuNzc0IDIzLjIgMCAyMi4yMjIgMGguMDAzeiIvPjwvc3ZnPgo=&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/aeon-toolkit/) [![!medium](https://img.shields.io/static/v1?logo=medium&label=Medium&message=blog&color=darkblue)](https://medium.com/@aeon.toolkit)   |
| **Affiliation** | [![numfocus](https://img.shields.io/badge/NumFOCUS-Affiliated%20Project-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org/sponsored-projects/affiliated-projects)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

<h4>Supported by</h4>
<p>
  <a href="https://www.inria.fr/">
    <img src="https://raw.githubusercontent.com/aeon-toolkit/aeon/main/docs/images/funder_logos/inria.png" alt="INRIA" height="50">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://www.ukri.org/councils/epsrc/">
    <img src="https://raw.githubusercontent.com/aeon-toolkit/aeon/main/docs/images/funder_logos/epsrc.png" alt="UKRI" height="50">
</a>
</p>

## From paper to `pip install`

`aeon` is developed in close contact with the time series research community.
Many of its algorithms are contributed or maintained by their original authors,
and the same team behind `aeon` runs the [benchmarks](https://timeseriesclassification.com/) that the field uses to
evaluate new methods. That means:

- **Faithful implementations.** Algorithms reflect what the papers actually describe.
- **State of the art, sooner.** New methods often land in `aeon` alongside publication.
- **Evidence-based defaults.** What's included — and what's recommended — is grounded in published comparative studies.


A selection of algorithms available in `aeon` written by ``aeon`` core developers or contributors:

| Method                 | Reference                                                                                 | Task                   |
|------------------------|-------------------------------------------------------------------------------------------|------------------------|
| **InceptionTime**      | [Ismail-Fawaz et al., 2020](https://link.springer.com/article/10.1007/s10618-020-00710-y) | Classification         |
| **Hydra-MultiRocket**  | [Dempster et al., 2023](https://link.springer.com/article/10.1007/s10618-023-00939-3)     | Classification         |
| **SETAR-Tree**         | [Godahewa et al., 2023](https://link.springer.com/article/10.1007/s10994-023-06316-x)     | Forecasting            |
| **KASBA**              | [Holder et al., 2026](https://link.springer.com/article/10.1007/s10618-026-01189-9)       | Clustering             |
| **CLASP**              | [Ermshaus et al., 2023](https://link.springer.com/article/10.1007/s10618-023-00923-x)     | Segmentation           |
| **DrCIF**              | [Guijo-Rubio et al., 2024](https://link.springer.com/article/10.1007/s10618-024-01027-w)  | Regression             |
| **TDE**                | [Guijo-Rubio et al., 2025](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10769513) | Ordinal Classification |

Code in `aeon` and related toolkits has been used in a wide range of benchmarking studies


| Task                               | Reference                                                                                 | Task           |
|------------------------------------|-------------------------------------------------------------------------------------------|----------------|
| Multivariate classification        | [Middlehurst et al., 2026](https://arxiv.org/abs/2603.20352)                              | Classification |
| Clustering                         | [Holder et al., 2024](https://link.springer.com/article/10.1007/s10115-023-01952-0)       | Benchmarking   |
| Classification (the "bake off")    | [Bagnall et al., 2017](https://link.springer.com/article/10.1007/S10618-016-0483-9)       | Benchmarking   |
| Classification ("bake off redux")  | [Middlehurst et al., 2025](https://link.springer.com/article/10.1007/s10618-024-01022-1)  | Benchmarking   |
| Deep learning for classification   | [Ismail-Fawaz et al., 2019](https://link.springer.com/article/10.1007/s10618-019-00619-1) | Benchmarking   |


See the [API reference](https://www.aeon-toolkit.org/en/stable/api_reference.html)
for the full list of estimators across all tasks.

⭐ **Star the repo** to follow new releases — `aeon` ships frequently, and starring is the easiest way to know when new algorithms arrive.


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
Fit a classifier on a standard UCR dataset:
```python
from aeon.classification.convolution_based import RocketClassifier
from aeon.datasets import load_gunpoint

X_train, y_train = load_gunpoint(split="train")
X_test, y_test = load_gunpoint(split="test")

clf = RocketClassifier()
clf.fit(X_train, y_train)

print("Accuracy:", clf.score(X_test, y_test))
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

clf = MultiRocketHydraClassifier(n_kernels=100)
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
## Deep learning for time series

`aeon` provides Keras/TensorFlow implementations of leading deep learning
architectures for time series through the `networks` module, with a consistent scikit-learn compatible API
and many models contributed by their original authors:

- **Classification:** InceptionTime, H-InceptionTime, LITE, LITETime, ResNet, FCN, MLP, CNN, Disjoint-CNN, and more
- **Regression:** the same backbone architectures, adapted for continuous targets
- **Clustering:** deep learning based clustering via learned representations
- **Forecasting:** deep learning based forecasting

A minimal example:

```python
from aeon.datasets import load_basic_motions
from aeon.classification.deep_learning import InceptionTimeClassifier

X_train, y_train = load_basic_motions(split="train")
X_test, y_test = load_basic_motions(split="test")

clf = InceptionTimeClassifier(n_epochs=10)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
```

### Clustering

Time series clustering groups similar time series together from an unlabelled collection.

```python
import numpy as np
from aeon.clustering import KASBA
from aeon.datasets import load_gunpoint

X, y = load_gunpoint()
clu = KASBA(n_clusters=2)
clu.fit(X)

print(clu.labels_)
```

### Forecasting

`aeon` provides a wide range of forecasting algorithms, including classic
statistical models and modern deep learning approaches.

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

For project or collaboration enquiries, contact **[contact@aeon-toolkit.org](mailto:contact@aeon-toolkit.org)**.

## Support aeon

There are several ways to engage with the project:

- ⭐ **Star this repository** to follow new releases
- 📝 **[Cite](https://github.com/aeon-toolkit/aeon/blob/main/CITATION.cff) `aeon`** in academic work if you use it
- 🐛 **Report bugs or request features** via [GitHub Issues](https://github.com/aeon-toolkit/aeon/issues)
- 💬 **Ask questions or join the discussion** on [GitHub Discussions](https://github.com/aeon-toolkit/aeon/discussions) or [Discord](https://discord.gg/D6rzqHGKRJ)
- 🛠️ **Contribute** code, tests, documentation, or examples — see the [contributing guide](https://github.com/aeon-toolkit/aeon/blob/main/CONTRIBUTING.md)

For project or collaboration enquiries, contact **[contact@aeon-toolkit.org](mailto:contact@aeon-toolkit.org)**.

## Contributing to aeon

If you are interested in contributing, please read the
[contributing guide](https://github.com/aeon-toolkit/aeon/blob/main/CONTRIBUTING.md)
before opening a pull request or taking ownership of an issue.

Useful links:

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

If you let us know about your paper using `aeon`, we will happily list it on the [project website](https://www.aeon-toolkit.org/en/latest/papers_using_aeon.html).

## Project history

`aeon` was forked from `sktime` `v0.16.0` in 2022 by an initial group of eight core developers, and has since been substantially rewritten and extended.
Our core development team of 13 spans academia and industry, representing seven nationalities across the globe.
You can read more about the project's history, values, and governance on the [About Us page](https://www.aeon-toolkit.org/en/stable/about.html).

## Project status

`aeon` is under active development. The core package is stable and widely used. The following modules are currently considered in development, and the deprecation policy does not necessarily apply (although we only rarely make non-compatible changes): `anomaly_detection`, `forecasting`, `segmentation`, `similarity_search`, `visualisation`, `transformations.collection.self_supervised`, `transformations.collection.imbalance`.

Please check the documentation for task-specific capabilities, limitations, and current status.
