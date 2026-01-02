
[//]: # (Try to put references in harvard style for consistency.)

# aeon Projects

`aeon` runs a range of short to medium duration projects that involve
developing or using `aeon` and interacting with the community and the code
base. These projects are designed for internships, usage as part of
undergraduate/postgraduate projects at academic institutions, options for
programs such as [Google Summer of Code (GSoC)](https://summerofcode.withgoogle.com/) or just for personal side projects.
For those interested in undertaking a project outside these scenarios, we recommend
joining the [Slack](https://join.slack.com/t/aeon-toolkit/shared_invite/zt-3ihx5vif8-SwFzy1unNNMeQueC84MXVA)and discussing with the community.

Feel free to propose your own project ideas, but please discuss them with us first.
We have an active community of researchers and students who work on `aeon`.
Please get in touch via Slack if you are interested in any of these projects or have
any questions. We will more widely advertise funding opportunities as and when they
become available.

All the projects listed will require knowledge of Python and Git/GitHub. The
majority of them will require some knowledge of machine learning and time series.

## Current `aeon` projects

This is a list of some of the projects we are interested in running (last updated
25/05/2025):

[Classification](#classification)
1. Optimizing the Shapelet Transform for classification and similarity search
2. Improved HIVE-COTE implementation
3. Compare distance-based classification.

[Clustering](#clustering)
1. Density peaks clustering algorithm
2. Hierarchical clustering for time series

[Transformation](#transformation)
1. Improve ROCKET family of transformers

[Visualisation](#visualisation)
1. Explainable AI with the shapelet transform

[Documentation](#documentation)
1. Improve automated API documentation
2. Improve the documentation tag interactivity and testing

[Maintenance](#maintenance)
1. Modernising the `aeon` linting and type checking workflows

[Multi-module](#multi-module)
1. Implementing multithreading for `aeon` estimators and tools for evaluating multithreading performance

## Classification

### 1. Optimizing the Shapelet Transform for Classification and Similarity Search

Contact: Antoine Guillaume ({user}`baraline`) and Tony Bagnall ({user}`TonyBagnall`)

#### Related Issues

[#186](https://github.com/aeon-toolkit/aeon/issues/186)
[#973](https://github.com/aeon-toolkit/aeon/issues/973)
[#1322](https://github.com/aeon-toolkit/aeon/issues/1322)

#### Description

A shapelet is defined as a time series subsequence representing a pattern of interest
that we wish to search for in time series data. Shapelet-based algorithms can be used
for a wide range of time series tasks. In this project, we will focus on its core
application, which is to create an embedding of the input time series.

Our goal in this project will be to optimize the code related to the shapelet
transform method, which takes as input a set of shapelets and a time series dataset,
and give as output a tabular dataset containing the features characterizing the
presence (or absence) of each shapelet in the input time series (more information
in [1] and [2]).

Similarity search is another field of time series, which has proposed greatly optimized
algorithms (see [3] and [4]) for the task of finding the best matches of a subsequence
inside another time series. As this task is extremely similar to what is done in the
shapelet transform, we want to adapt these algorithms to the context of shapelets,
in order to achieve significant speed-ups.

#### Project stages

To achieve this goal, with the assistance of the mentor, we identify the following
steps for the mentee:

1. Learn about aeon best practices, coding standards and testing policies.
2. Study the shapelet transform algorithm and how it is related to the task of
similarity search.
3. Study the similarity search algorithms for the Euclidean distance and the
computational optimization they use.
4. Propose a candidate implementation for to increase the performance of the
computations made by a single shapelet. This can be made with the help of the existing
implementation of the similarity search module in `aeon`.
5. Measure the performance of this first candidate implementation against the current
approach.
6. Implement this solution to the shapelet transform algorithm, which uses multiple
shapelets.
7. Benchmark the implementation against the original shapelet transform algorithm.
8. If time, generalize this new algorithm to the case of dilated shapelets (see [5]).

#### References

1. Hills, J., Lines, J., Baranauskas, E., Mapp, J. and Bagnall, A., 2014.
Classification of time series by shapelet transformation. Data mining and knowledge
discovery, 28, pp.851-881.
2. Bostrom, A. and Bagnall, A., 2017. Binary shapelet transform for multiclass time
series classification. Transactions on Large-Scale Data-and Knowledge-Centered Systems
XXXII: Special Issue on Big Data Analytics and Knowledge Discovery, pp.24-46.
3. Yeh, C.C.M., Zhu, Y., Ulanova, L., Begum, N., Ding, Y., Dau, H.A., Silva, D.F.,
Mueen, A. and Keogh, E., 2016, December. Matrix profile I: all pairs similarity joins
for time series: a unifying view that includes motifs, discords and shapelets. In 2016
IEEE 16th international conference on data mining (ICDM) (pp. 1317-1322). Ieee.
4. Zhu, Y., Zimmerman, Z., Senobari, N.S., Yeh, C.C.M., Funning, G., Mueen, A., Brisk,
P. and Keogh, E., 2016, December. Matrix profile ii: Exploiting a novel algorithm and
gpus to break the one hundred million barrier for time series motifs and joins. In 2016
IEEE 16th international conference on data mining (ICDM) (pp. 739-748). IEEE.
5. Guillaume, A., Vrain, C. and Elloumi, W., 2022, June. Random dilated shapelet
transform: A new approach for time series shapelets. In International Conference on
Pattern Recognition and Artificial Intelligence (pp. 653-664). Cham: Springer
International Publishing.

### 2. Improved HIVE-COTE implementation

Contact: Matthew Middlehurst ({user}`MatthewMiddlehurst`) and Tony Bagnall
({user}`TonyBagnall`)

#### Related Issues

[#663](https://github.com/aeon-toolkit/aeon/issues/663)
[#1646](https://github.com/aeon-toolkit/aeon/issues/1646)

#### Description

The Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) [1,2,3]
is a time series classifier that has claims to be state-of-the-art,
particularly in terms of probabilistic estimates [4]. There have been several iterations
that use different base classifiers but they all share the same design basic:
classifiers using different representations are combined in a weighted meta-ensemble
[4]. There are two HIVE-COTE implementations currently in `aeon`: HIVE-COTEV1 [2]
and HIVECOTEV2 [3]. This project will involve combining these into a single
framework, modularising the ensemble stage and possibly experimenting with
alternative structures.

#### Project stages

1. Learn about `aeon` best practices, coding standards and testing policies.
2. Study the HIVE-COTE algorithm and previous `aeon` implementation.
3. Implement a modular HIVE-COTE or composable classification ensemble framework.
4. Convert HIVE-COTEV1 and HIVE-COTEV2 classifiers to use this framework ensuring
   results remain the same.
5. Restructure the ensemble stage to allow easy experimentation with variants.

#### References

1. A Bagnall, J Lines, J Hills, A Bostrom, Time-series classification with COTE: the
   collective of transformation-based ensembles, IEEE Transactions on Knowledge and
   Data Engineering 27 (9), 2522-2535
2. M Middlehurst, J Large, M Flynn, J Lines, A Bostrom, A Bagnall, HIVE-COTE 2.0: a
   new meta ensemble for time series classification, Machine Learning 110 (11),
   3211-3243
3. J Lines, S Taylor, A Bagnall, Time series classification with HIVE-COTE: The
   hierarchical vote collective of transformation-based ensembles, ACM Transactions
   on Knowledge Discovery from Data (TKDD) 12 (5), 1-35
4. Middlehurst, M., Schäfer, P. and Bagnall, A., 2024. Bake off redux: a review and
   experimental evaluation of recent time series classification algorithms. Data Mining
   and Knowledge Discovery, 38(4), pp.1958-2031.

### 3. Compare distance-based classification and regression

Contact: Chris Holder ({user}`chrisholder`) and Tony Bagnall
({user}`TonyBagnall`)

#### Related Issues

[#424](https://github.com/aeon-toolkit/aeon/issues/424)
[#425](https://github.com/aeon-toolkit/aeon/issues/425)
[#426](https://github.com/aeon-toolkit/aeon/issues/426)
[#427](https://github.com/aeon-toolkit/aeon/issues/427)
[#488](https://github.com/aeon-toolkit/aeon/issues/488)

#### Description

Distance-based algorithms are popular for time series classification and regression.
However, the evaluation of distance functions for classification have not
comprehensively covered all possible uses. For example, there has not been a proper
bake off for using elastic distance with support vector machines or with tuning
distance functions and classifiers in combination. This project will combine
implementing alternative distance functions and comparing performance on the UCR
datasets.

#### Project stages

1. Learn about `aeon` best practices, coding standards and testing policies.
2. Study the distance model and the distance-based classifiers in `aeon`.
3. Read and implement alternative distance functions in the format of the
   `aeon.distances` module.
4. Test the distances against previous results if available.
5. Run a mini-bakeoff of distance-based classifiers on the UCR datasets, comparing
   distances using existing distance-based classifiers in `aeon` and `scikit-learn`.

## Clustering

### 1. Density peaks clustering algorithm

Contact: Tony Bagnall ({user}`TonyBagnall`) and Chris Holder (`@chrisholder`).

#### Description

The clustering module in `aeon`, up until now, clusters using time series
specific distance functions with partitional clustering algorithms such as k-means and
k-medoids. An alternative clustering algorithm is density peaks [1]. This clustering
algorithm has the benefit of not having to label all cases as cluster members, which
means it can easily be adapted to anomaly detection [2]. It is
a general purpose clustering algorithm that is not available in scikit learn. This
project will implement the algorithm based on Java and matlab implementations then
compare performance against partitional clustering for time series clustering.

#### Project Stages

1. Research and understand how density peaks works.
2. Implement density peaks as an `aeon` estimator using the `aeon` distances module.
3. Test the implementation against other implementations for correctness and on
   synthetic data used in the publication.
4. Compare against alternative TSCL algorithms.
5. Possible extensions to reflect recent research [2] with specific time series
   components [3].

#### References

1. Rodriguez, A., & Laio, A. Clustering by Fast Search and Find of
   Density Peaks. [Science](https://www.science.org/doi/10.1126/science.1242072), 344 (6191), 1492-1496, 2014.
2. Chen, L., Gao, S. & Liu, B. An improved density peaks clustering algorithm
   based on grid screening and mutual neighborhood degree for network anomaly detection.
   Sci Rep 12, 1409 (2022) [DOI](https://doi.org/10.1038/s41598-021-02038-z)
3. Begum et al. A General Framework for Density Based  Time Series Clustering
   Exploiting a Novel Admissible Pruning Strategy, [arXiv](https://arxiv.org/ftp/arxiv/papers/1612/1612.00637.pdf)

### 2. Hierarchical clustering for time series

Contact: Tony Bagnall ({user}`TonyBagnall`) and Chris Holder (`@chrisholder`).

#### Description

While the aeon distances module is already extensive, there are still clusterers that
could be implemented. aeon currently has common algorithms such as KMeans and KMedoids,
but is missing Hierarchical clustering approaches. The project will involve implementing
and evaluating some of these, and ensuring they are properly integrated to use the
wide variety of functions in the distances module.

#### Project Stages

1. Research and understand hierarchical clustering algorithms work and different
   methods.
2. Implement hierarchical clustering algorithms as an `aeon` estimator using the
   `aeon` distances module.
3. Implement a dendrogram visualisation for the clustering.
4. Test the implementation visualisations and results against other implementations
   for correctness

## Transformation

### 1. Improve ROCKET family of transformers

Contact: Ali Ismail-Fawaz ({user}`hadifawaz1999`) and Matthew Middlehurst
({user}`MatthewMiddlehurst`)

#### Related Issues

[#313](https://github.com/aeon-toolkit/aeon/issues/313)
[#1126](https://github.com/aeon-toolkit/aeon/issues/1126)
[#1248](https://github.com/aeon-toolkit/aeon/issues/1248)
[#2179](https://github.com/aeon-toolkit/aeon/issues/2179)

#### Description

The ROCKET algorithm [1] is a very fast and accurate transformation designed for time
series classification. It is based on a randomly initialised convolutional kernels that
are applied to the time series and used to extract summary statistics. ROCKET has
applications to time series classification, extrinsic regression and anomaly detection,
but as a fast and unsupervised transformation, it has potential to a wide range of
other time series tasks.

`aeon` has implementations of the ROCKET transformation and its variants, including
MiniROCKET [2] and MultiROCKET [3]. However, these implementations have room for
improvement. There is scope to speed up the implementations, and the amount of variants
is likely unnecessary and could be condensed into higher quality estimators.

This projects involves improving the existing ROCKET implementations in `aeon` or
implementing new ROCKET variants. The project will involve benchmarking to ensure that
the new implementations are as fast and accurate as the original ROCKET algorithm and
potentially to compare to other implementations.
Besides improving the existing implementations, there is scope to implement a
probabilistic ridge classifier [4] for the algorithms to use or implement GPU compatible
versions of the algorithms.

#### Project Stages

1. Learn about `aeon` best practices, coding standards and testing policies.
2. Study the ROCKET, MiniROCKET, MultiROCKET algorithms and existing
   implementations in `aeon`.
3. Merge and tidy the ROCKET implementations, with the aim being to familiarise the
mentee with the `aeon` pull request process.
4. Implement one (or more) of the proposed ROCKET implementation improvements:
   * Significantly alter the current ROCKET implementations with the goal of
   speeding up the implementation on CPU processing.
   * Implement a GPU version of some of the ROCKET transformers, using either
   `tensorflow` or `pytorch`.
   * Implement probabilistic ridge classifier as a `scikit-learn` estimator.
5. Benchmark the implementation against the original ROCKET implementations, looking at
booth speed of the transform and accuracy in a classification setting.

#### References

1. Dempster, A., Petitjean, F. and Webb, G.I., 2020. ROCKET: exceptionally fast and
accurate time series classification using random convolutional kernels.
Data Mining and Knowledge Discovery, 34(5), pp.1454-1495.
2. Dempster, A., Schmidt, D.F. and Webb, G.I., 2021, August. Minirocket: A very fast
(almost) deterministic transform for time series classification. In Proceedings of the
27th ACM SIGKDD conference on knowledge discovery & data mining (pp. 248-257).
3. Tan, C.W., Dempster, A., Bergmeir, C. and Webb, G.I., 2022. MultiRocket: multiple
pooling operators and transformations for fast and effective time series classification.
Data Mining and Knowledge Discovery, 36(5), pp.1623-1646.
4. Dempster, A., Webb, G.I. and Schmidt, D.F., 2024. Prevalidated ridge regression is a
highly-efficient drop-in replacement for logistic regression for high-dimensional data.
arXiv preprint arXiv:2401.15610.

## Visualisation

### 1. Explainable AI with the shapelet transform.

Contact: TonyBagnall ({user}`TonyBagnall`) and David Guijo-Rubio
({user}`dguijo`)

#### Description

This project will focus on explainable AI for time series classification (TSC) [1],
specifically the family of algorithms based on shapelets [2]. Shapelets are small sub
 certain shape of heartbeat, perhaps a short irregularity, might be useful in
predicting the medical condition. We will look at the shapelet transform classifier
[3]. This finds a large set of shapelets from the training data and uses them to
build a classifier. We want to develop tools to help us visualise the output of the
search for good shapelets to help explain why predictions are made. This project is
not tied to a specific data set. It is to develop tools to help any user of the
toolkit.  It will involve learning about aeon and making contributions to open
source toolkits, familiarisation with the shapelet code and the development of a
visualisation tool to help relate shapelets back to the training data. An outline
for the project is

#### Project Stages

1. Familiarisation with open source, aeon and the visualisation module. Make
contribution for a good first issue.
2. Understand the shapelet transfer algorithm, engage in ongoing discussions for
possible improvements, run experiments to create predictive models for a test data set
3. Design and prototype visualisation tools for shapelets, involving a range
of summary measures and visualisation techniques, including plotting shapelets on
training data, calculating frequency, measuring similarity between
4. Debug, document and make PRs to merge contributions into the aeon toolkit.

#### References

1. Bagnall, A., Lines, J., Bostrom, A., Large, J. and Keogh, E. The great time series
classification bake off: a review and experimental evaluation of recent algorithmic
advances. Data Mining and Knowledge Discovery, Volume 31, pages 606–660, (2017)
2. Ye, L., Keogh, E. Time series shapelets: a novel technique that allows accurate,
interpretable and fast classification. Data Min Knowl Disc 22, 149–182 (2011).
https://doi.org/10.1007/s10618-010-0179-5
3. Lines, L., Davis, L., Hills, J. and Bagnall, A. A shapelet transform for time
series classification, KDD '12: Proceedings of the 18th ACM SIGKDD international
conference on Knowledge discovery and data mining (2012)
https://doi.org/10.1145/2339530.2339579

## Documentation

### 1. Improve the documentation codebase interactions and testing

Contact: Matthew Middlehurst ({user}`MatthewMiddlehurst`)

#### Description

The `aeon` documentation is a key resource for users of the toolkit. It provides
information on how to install the toolkit, how to use the toolkit, and how to
contribute to the toolkit. The `aeon` documentation is built using `sphinx` and hosted
on `readthedocs`.

While there are always improvements that can be made to the general documentation itself
(e.g., improving the clarity of the text, adding more examples, etc.) for both webpages
and estimator docstrings, this project focuses on implementing functions to
automatically link relevant API pages together and ensure new pull requests are
accompanied by the appropriate documentation. Some examples of improvements that could
be made include:

- Linking to examples to in API pages where the function/class is used
- Improving the [estimator overview page](https://www.aeon-toolkit.org/en/stable/estimator_overview.html)
by further integrating the tags system or adding search and filtering functionality
- Implementing workflows to ensure that new public functionality includes a valid
docstring (i.e. has a description, parameters, returns, etc. sections where relevant)

There is a lot of potential for additional functionality, so feel free to suggest
improvements or new features outside the examples provided.

#### Project Stages

1. Learn about `aeon` best practices and project documentation.
2. Familiarise with `sphinx` documentation generation and `numpydoc` docstring
standards.
3. Improve the API documentation for a few classes/functions and go through the Pull
Request and review process.
4. Implement at least one new feature or improvement to the aeon documentation webpage
(outside of general text improvements) And/Or improve the aeon testing suite to ensure
that new PRs are accompanied by the appropriate documentation.

## Maintenance

### Modernising the `aeon` linting and type checking workflows

Contact: Matthew Middlehurst ({user}`MatthewMiddlehurst`)

#### Description

This project involves updating the `aeon` linting and type checking workflows to
use modern tools and ensure that the codebase is up to date with the latest
Python standards.

The `aeon` toolkit uses `pre-commit` to run code quality checks on all code changes
and ensure that they meet the project's standards. This includes a number of checks and
formatting tools, such as `black`, `flake8`, and `isort` (see [here](https://github.com/aeon-toolkit/aeon/blob/main/.pre-commit-config.yaml)).
Over time new tools have been released such as `ruff` and tools we previously used such
as `pydocstyle` have been deprecated. The first part of this project will involve
modernising the `pre-commit` configuration to use the latest tools.

`aeon` contributors have been encouraged to add type hints to the codebase, but this
is a gradual process and there are still many parts of the codebase that are not fully
typed. A big issue we face in this is the current lack of automated testing to ensure
that implemented type hints are accurate. This second part project will involve
implementing robust testing utilities to help contributors and reviewers ensure that
new type hints are correct.

Other ideas to improve the code quality testing in `aeon` pull requests or deliver
feedback from tests to contributors are welcome.

#### Expected Outcome(s)

1. Learn about used for code quality checks and type checking in Python.
2. Familiarise yourself with the `aeon` CI including pre-commit and GitHub Actions
workflows.
3. Update workflows for checking code quality in `aeon` pull requests
4. Implement automated testing and utilities to help contributors implement accurate
type hints for `aeon` code.

## Multi-module

### Implementing multithreading for `aeon` estimators and tools for evaluating multithreading performance

Contact: Matthew Middlehurst ({user}`MatthewMiddlehurst`)

#### Description

Multithreading in `aeon` for estimators does not have a set structure or library that must
be used. Most algorithms which have an `n_jobs` parameter available use a mix of `Joblib` and
`numba` multithreading. Algorithms which do have the capability for multithreading have not
been thoroughly tested, and as such the efficiency of these implementations is unknown.

As well as expanding the amount of estimators which can use multiple threads, we would
like to develop tools to evaluate whether this threading is efficient and develop
documentation for contributors which want to add multithreading to `aeon` estimators.

#### Project Stages

1. Investigate Python multithreading libraries and `numba` multithreading.
2. Learn about `aeon` best practices, coding standards and how current estimators use
multiple threads.
3. Write tools and testing for evaluating the efficiency of multithreaded code while
maintaining single-threaded performance.
4. If any estimators are performing poorly with multithreading, implement
improvements to the multithreading implementation.
5. Implement multithreading as a capability for currently lacking `aeon` estimators
(preferably in the `classification`, `regression` and `clustering` modules to start).


```{toctree}
:hidden:

projects/previous_projects.md
```
