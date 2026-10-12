---
html_theme.sidebar_secondary.remove: true
---

# A toolkit for time series machine learning

:::{div} aeon-lead
`aeon` brings classification, regression, clustering, anomaly detection, forecasting,
segmentation and similarity search for time series into one library. It follows the
`scikit-learn` interface, so its estimators work with the tools you already use.
:::

::::{div} aeon-hero-actions

```bash
pip install aeon
```

```{button-ref} getting_started
:ref-type: doc
:color: primary

Get started
```

```{button-ref} installation
:ref-type: doc
:color: primary
:outline:

Installation options
```

::::

```{raw} html
:file: images/landing/hero.html
```

## What you can do with aeon

Twelve modules cover learning from time series, transforming and comparing them, and
loading data and evaluating results. Each one comes with an example notebook.

::::{grid} 2 2 3 4
:gutter: 3
:class-container: aeon-tasks

:::{grid-item-card} Classification
:link: examples/classification/classification
:link-type: doc
:shadow: none

```{raw} html
:file: images/landing/icons/classification.svg
```
^^^
Predict a class label for each series.
:::

:::{grid-item-card} Regression
:link: examples/regression/regression
:link-type: doc
:shadow: none

```{raw} html
:file: images/landing/icons/regression.svg
```
^^^
Predict a continuous value for each series.
:::

:::{grid-item-card} Clustering
:link: examples/clustering/clustering
:link-type: doc
:shadow: none

```{raw} html
:file: images/landing/icons/clustering.svg
```
^^^
Group similar series without labels.
:::

:::{grid-item-card} Anomaly detection {bdg-secondary-line}`experimental`
:link: examples/anomaly_detection/anomaly_detection
:link-type: doc
:shadow: none

```{raw} html
:file: images/landing/icons/anomaly_detection.svg
```
^^^
Find unusual points or subsequences in a series.
:::

:::{grid-item-card} Forecasting {bdg-secondary-line}`experimental`
:link: examples/forecasting/forecasting
:link-type: doc
:shadow: none

```{raw} html
:file: images/landing/icons/forecasting.svg
```
^^^
Predict the future values of a series.
:::

:::{grid-item-card} Segmentation {bdg-secondary-line}`experimental`
:link: examples/segmentation/segmentation
:link-type: doc
:shadow: none

```{raw} html
:file: images/landing/icons/segmentation.svg
```
^^^
Split a series into regions that behave differently.
:::

:::{grid-item-card} Similarity search {bdg-secondary-line}`experimental`
:link: examples/similarity_search/similarity_search
:link-type: doc
:shadow: none

```{raw} html
:file: images/landing/icons/similarity_search.svg
```
^^^
Find the closest matches to a query in a collection of series.
:::

:::{grid-item-card} Transformations
:link: examples/transformations/transformations
:link-type: doc
:shadow: none

```{raw} html
:file: images/landing/icons/transformations.svg
```
^^^
Extract features from series or change their representation.
:::

:::{grid-item-card} Distances
:link: examples/distances/distances
:link-type: doc
:shadow: none

```{raw} html
:file: images/landing/icons/distances.svg
```
^^^
Measure how far apart two series are, with elastic distances such as DTW.
:::

:::{grid-item-card} Networks
:link: examples/networks/deep_learning
:link-type: doc
:shadow: none

```{raw} html
:file: images/landing/icons/networks.svg
```
^^^
Deep learning architectures for time series.
:::

:::{grid-item-card} Data
:link: examples/datasets/datasets
:link-type: doc
:shadow: none

```{raw} html
:file: images/landing/icons/datasets.svg
```
^^^
The data structures used in `aeon` and how to load datasets.
:::

:::{grid-item-card} Benchmarking
:link: examples/benchmarking/benchmarking
:link-type: doc
:shadow: none

```{raw} html
:file: images/landing/icons/benchmarking.svg
```
^^^
Compare algorithms and reproduce published results.
:::

::::

## Works like scikit-learn

If you have trained a `scikit-learn` model, you already know how to use `aeon`.

::::{grid} 1 1 2 2
:gutter: 5
:class-container: aeon-code-example

:::{grid-item}
:columns: 12 12 5 5
:class: aeon-steps

1. Load a collection of time series. Here, one day of electricity demand per series,
   labelled by season.
2. Choose an estimator. `RocketClassifier` is a fast and accurate place to start.
3. Call `fit`, `predict` and `score`, as you would with any `scikit-learn` estimator.

The same estimators work inside pipelines, cross-validation and grid search.
:::

:::{grid-item}
:columns: 12 12 7 7

```python
from aeon.classification.convolution_based import RocketClassifier
from aeon.datasets import load_italy_power_demand

X_train, y_train = load_italy_power_demand(split="train")
X_test, y_test = load_italy_power_demand(split="test")

clf = RocketClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
```

:::

::::

## Why aeon

::::{grid} 1 2 2 4
:gutter: 4
:class-container: aeon-reasons

:::{grid-item}
<span class="aeon-reason-icon"><i class="fa-solid fa-medal"></i></span>

**State of the art**

We provide a broad library of time series algorithms, including the latest advances
for many tasks.
:::

:::{grid-item}
<span class="aeon-reason-icon"><i class="fa-solid fa-gauge-high"></i></span>

**Fast**

Our algorithms are implemented as efficiently as possible, for example by using
`numba`.
:::

:::{grid-item}
<span class="aeon-reason-icon"><i class="fa-solid fa-puzzle-piece"></i></span>

**Compatible**

`aeon` is built on top of `scikit-learn`, so it integrates with other machine
learning libraries and time series packages.
:::

:::{grid-item}
<span class="aeon-reason-icon"><i class="fa-solid fa-flask"></i></span>

**Reproducible**

We provide tools to reproduce benchmarking results and to evaluate time series
algorithms from `aeon` and other `scikit-learn` compatible packages.
:::

::::

## Join the community

`aeon` is developed in the open by volunteers. Questions, bug reports and new
contributors are all welcome. You can also write to
[contact@aeon-toolkit.org](mailto:contact@aeon-toolkit.org).

::::{grid} 1 2 2 4
:gutter: 3
:class-container: aeon-community

:::{grid-item-card}
:link: https://discord.gg/D6rzqHGKRJ
:link-alt: aeon on Discord
:shadow: none

<span class="aeon-community-icon"><i class="fa-brands fa-discord"></i></span>

**Discord**
<span class="aeon-community-text">Ask questions and talk to the developers</span>
:::

:::{grid-item-card}
:link: https://github.com/aeon-toolkit/aeon
:link-alt: aeon on GitHub
:shadow: none

<span class="aeon-community-icon"><i class="fa-brands fa-github"></i></span>

**GitHub**
<span class="aeon-community-text">Report bugs and contribute code</span>
:::

:::{grid-item-card}
:link: https://www.linkedin.com/company/aeon-toolkit
:link-alt: aeon on LinkedIn
:shadow: none

<span class="aeon-community-icon"><i class="fa-brands fa-linkedin"></i></span>

**LinkedIn**
<span class="aeon-community-text">Follow releases and project news</span>
:::

:::{grid-item-card}
:link: https://medium.com/@aeon.toolkit
:link-alt: aeon on Medium
:shadow: none

<span class="aeon-community-icon"><i class="fa-brands fa-medium"></i></span>

**Medium**
<span class="aeon-community-text">Read articles from the developers</span>
:::

::::

## Experimental modules

Some modules of `aeon` are still experimental and may have changing interfaces.
To support development on these modules, the [deprecation policy](developer_guide/deprecation.md)
is relaxed, so it is suggested that you integrate these modules with care. The current
experimental modules are:

- `anomaly_detection`
- `forecasting`
- `segmentation`
- `similarity_search`
- `visualisation`
- `transformations.collection.self_supervised`
- `transformations.collection.imbalance`

```{toctree}
:hidden:

installation.md
getting_started.md
api_reference.md
examples.md
Estimators <estimator_overview.md>
Development <contributing.md>
About <about.md>
changelog.md
```
