
[//]: # (Try to put references in harvard style for consistency.)

# Mentoring and Projects

`aeon` runs a range of short projects interacting with the community and the code
base. These projects are designed for internships, usage as part of
undergraduate/postgraduate projects at academic institutions, and as options for
programs such as [Google Summer of Code (GSoC)](https://summerofcode.withgoogle.com/).

For those interested in undertaking a project outside these scenarios, we recommend
joining the [Slack](https://join.slack.com/t/aeon-toolkit/shared_invite/zt-22vwvut29-HDpCu~7VBUozyfL_8j3dLA)
and discussing with the project mentors. We aim to run schemes to
help new contributors to become more familiar with `aeon`, time series machine learning
research, and open-source software development.

All the projects listed will require knowledge of Python 3 and Git/GitHub. The
majority of them will require some knowledge of machine learning and time series.

## Current aeon projects

This is a list of some of the projects we are interested in running in 2024. Feel
free to propose your own project ideas, but please discuss them with us first. We have
an active community of researchers and students who work on `aeon`. Please get in touch
via Slack if you are interested in any of these projects or have any questions.

We will more widely advertise funding opportunities as and when they become available.

### Forecasting

#### 1. Machine Learning for Time Series Forecasting

Mentors: Tony Bagnall ({user}`TonyBagnall`) and TBC.

##### Description

This project will investigate algorithms for forecasting based on traditional machine
learning (tree based) and time series machine learning (transformation based). Note
this project will not involve deep learning based forecasting. It will involve
helping develop the `aeon` framework to work more transparently with ML algorithms,
evaluating regression algorithms already in `aeon`[1] for forecasting problems and
implementing at least one algorithm from the literature not already in aeon, such as
SETAR-Tree [3].

##### Project Stages

1. Learn about aeon best practices, coding standards and testing policies.
2. Work through existing forecasting workflow and experimental reproduction.
3. Adapt the M competition set up [2] for ML experimental framework to assess time
   series regression algorithms [1].
4. Implement a machine learning forecasting algorithm [3]

##### Expected Outcomes

1. Contributions to the aeon forecasting module.
2. Implementation of a machine learning forecasting algortihms.
3. Help write up results for a technical report/academic paper (depending on outcomes).

##### Skills Required

1. Python 3
2. Git and GitHub
3. Some machine learning and/or forecasting background (e.g. taught courses or
   practical experience)

##### References

1. Guijo-Rubio, D.,Middlehurst, M., Arcencio, G., Furtado, D. and Bagnall, A.
Unsupervised Feature Based Algorithms for Time Series Extrinsic Regression,
arXiv2305.01429, 2023
2. https://forecasters.org/resources/time-series-data/
3. Godahewa, R., Webb, G.I., Schmidt, D. et al. SETAR-Tree: a novel and accurate
tree algorithm for global time series forecasting. Mach Learn 112, 2555–2591 (2023).
https://link.springer.com/article/10.1007/s10994-023-06316-x

#### 2. Deep Learning for Time Series Forecasting

Mentors: Ali Ismail-Fawaz ({user}`hadifawaz1999`)

##### Description

Implement and evaluate some models from the literature, maybe benchmark them as well
to non-deep models

##### Project Stages

TBC

##### Expected Outcomes

TBC

##### References

TBC

### Classification

#### 1. Optimizing the Shapelet Transform for Classification and Similarity Search

Mentors : Antoine Guillaume ({user}`baraline`) and Tony Bagnall ({user}`TonyBagnall`)

##### Description

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

##### Project stages

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

##### Expected Outcomes

We expect the mentee engage with the aeon community and produce a performance games
for the
We Based on the benchmark of the different implementations, we will evaluate the
performance gains of the new shapelet transform and the success of this project.

##### References

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

#### 2. EEG classification with aeon-neuro

Mentors: Tony Bagnall ({user}`TonyBagnall`) and Aiden Rushbrooke

##### Description

EEG (Electroencephalogram) data are high dimensional time series that are used in
medical, psychology and brain computer interface research. For example, EEG are
used to detect epilepsy and to control decvices such as mice. There is a huge body
of work on analysing and learning from EEG, but there is a wide disparity of
tools, practices and systems used. This project will help members of the `aeon`
team who are currently researching techniques for EEG classification [1] and
developing an aeon sister toolkit, ``aeon-neuro`` [LINK](https://github.com/aeon-toolkit/aeon-neuro). We will work together to
improve the structure and documentation for aeon-neuro, help integrate the
toolkit with existing EEG toolkits such as NM [2], provide interfaces to standard data
formats such as BIDS [3] and help develop and assess a range of EEG classification
algorithms.

##### Project stages

1. Learn about aeon best practices, coding standards and testing policies.
2. Study the existing techniques for EEG classification.
3. Implement or wrap standard EEG processing algorithms.
4. Evaluate aeon classifiers for EEG problems.
5. Implement alternatives transformations for preprocessing EEG data.
6. Help write up results for a technical report/academic paper (depending on outcomes).

##### Expected Outcomes

We would expect a better documented and more integrated aeon-neuro toolkit with
better functionality and a wider appeal.

##### References

1. Aiden Rushbrooke, Jordan Tsigarides, Saber Sami, Anthony Bagnall,
Time Series Classification of Electroencephalography Data, IWANN 2023.
2. MNE Toolkit, https://mne.tools/stable/index.html
3. The Brain Imaging Data Structure (BIDS) standard, https://bids.neuroimaging.io/

#### 3. Improved Proximity Forest for classification

Mentors: Matthew Middlehurst ({user}`MatthewMiddlehurst`) and Tony Bagnall
({user}`TonyBagnall`)

##### Description

Distance-based classifiers such as k-Nearest Neighbours are popular approaches to time
series classification. They primarily use elastic distance measures such as Dynamic Time
Warping (DTW) to compare two series. The Proximity Forest algorithm [1] is a
distance-based classifier for time series. The classifier creates a forest of decision
trees, where the tree splits are based on the distance between time series using
various distance measures. A recent review of time series classification algorithms [2]
found that Proximity Forest was the most accurate distance-based algorithm of those
compared.

`aeon` previously had an implementation of the Proximity Forest algorithm, but it was
not as accurate as the original implementation (the one used in the study) and was
unstable on benchmark datasets. The goal of this project is to significantly overhaul
the previous implementation or completely re-implement Proximity Forest in `aeon` to
match the accuracy of the original algorithm. This will involve comparing against the
authors' Java implementation of the algorithm as well as alternate Python versions.
The mentors will provide results for both for alternative methods. While knowing
Java is not a requirement for this project, it could be beneficial.

Recently, the group which published the algorithm has proposed a new version of the
Proximity Forest algorithm, Proximity Forest 2.0 [3]. This algorithm is more accurate
than the original Proximity Forest algorithm, and does not currently have an
implementation in `aeon` or elsewhere in Python. If time allows, the project could also
involve implementing and evaluating the Proximity Forest 2.0 algorithm.

##### Project stages

1. Learn about `aeon` best practices, coding standards and testing policies.
2. Study the Proximity Forest algorithm and previous `aeon` implementation.
3. Improve/re-implement the Proximity Forest implementation in `aeon`, with
the aim being to have an implementation that is as accurate as the original algorithm,
while remaining feasible to run.
4. Evaluate the improved implementation against the original `aeon` Proximity Forest
and the authors' Java implementation.
5. If time, implement the Proximity Forest 2.0 algorithm and repeat the above
evaluation.

##### Expected Outcomes

We expect the mentee engage with the aeon community and produce a high quality
implementation of the Proximity Forest algorithm(s) that gets accepted into the toolkit.

##### References

1. Lucas, B., Shifaz, A., Pelletier, C., O’Neill, L., Zaidi, N., Goethals,
B., Petitjean, F. and Webb, G.I., 2019. Proximity forest: an effective and scalable
distance-based classifier for time series. Data Mining and Knowledge Discovery, 33(3),
pp.607-635.
2. Middlehurst, M., Schäfer, P. and Bagnall, A., 2023. Bake off redux: a review and
experimental evaluation of recent time series classification algorithms. arXiv preprint
arXiv:2304.13029.
3. Herrmann, M., Tan, C.W., Salehi, M. and Webb, G.I., 2023. Proximity Forest 2.0: A
new effective and scalable similarity-based classifier for time series. arXiv
preprint arXiv:2304.05800.

### Clustering

#### 1. Feature based or deep learning based algorithms

Mentors: Tony Bagnall ({user}`TonyBagnall`), Ali Ismail-Fawaz ({user}`hadifawaz1999`)
and @Chris?

##### Description

Implement and evaluate some of the recently proposed clustering algorithms

The clustering module in `aeon`, up until now, primarily consists of distance-based
algorithms like K-Means, K-Medoids, and Clara, among others. Recently, we introduced an
initial deep clustering module featuring an FCN auto-encoder, incorporating
distance-based algorithms in the latent space. However, there is currently a shortage
of feature-based clustering algorithms.

The objective of this project is to enhance `aeon` by incorporating more deep learning
approaches for time series clustering. This involves adapting the FCN auto-encoder to
leverage the ResNet model. Additionally, the project aims to integrate feature-based
algorithms for time series clustering into the system.

##### Project Stages

TBC

##### Expected Outcomes

TBC

##### References

1. Lafabregue, B., Weber, J., Gançarski, P. and Forestier, G., 2022. End-to-end deep
representation learning for time series clustering: a comparative study. Data Mining
and Knowledge Discovery, 36(1), pp.29-81.

### Anomaly detection

#### 1. Anomaly detection with the Matrix Profile and MERLIN

Mentors: Matthew Middlehurst ({user}`MatthewMiddlehurst`)

##### Description

`aeon` is looking to extend its module for time series anomaly detection. The
end goal of this project is to implement the Matrix Profile [1][2] and MERLIN [3]
algorithms, but suitable framework for anomaly detection in `aeon` will need to be
designed first. The mentee will help design the API for the anomaly detection module
and implement the Matrix Profile and MERLIN algorithms.

Usage of external libraries such as `stumpy` [4] is possible for the algorithm
implementations, or the mentee can implement the algorithms from scratch using `numba`.
There is also scope to benchmark the implementations, but as there is no existing
anomaly detection module in `aeon`, this will require some infrastructure to be
developed and is subject to time and interest.

##### Project stages

1. Learn about `aeon` best practices, coding standards and testing policies.
2. Familiarise yourself with similar single series experimental modules in `aeon` such
as segmentation and similarity search.
3. Help design the API for the anomaly detection module.
4. Study and implement the Matrix Profile for anomaly detection and MERLIN algorithms
using the new API.
5. If time allows and there is interest, benchmark the implementations against the
original implementations or other anomaly detection algorithms.

##### Project Outcome

As the anomaly detection is a new module in `aeon`, there is very little existing code
to compare against and little infrastructure to evluate anomaly detection algorithms.
The success of the project will be evaluated by the quality of the code produced and
engagement with the project and the `aeon` community.

##### References

1. Yeh, C.C.M., Zhu, Y., Ulanova, L., Begum, N., Ding, Y., Dau, H.A., Silva, D.F.,
Mueen, A. and Keogh, E., 2016, December. Matrix profile I: all pairs similarity joins
for time series: a unifying view that includes motifs, discords and shapelets. In 2016
IEEE 16th international conference on data mining (ICDM) (pp. 1317-1322). Ieee.
2. Lu, Y., Wu, R., Mueen, A., Zuluaga, M.A. and Keogh, E., 2022, August.
Matrix profile XXIV: scaling time series anomaly detection to trillions of datapoints
and ultra-fast arriving data streams. In Proceedings of the 28th ACM SIGKDD Conference
on Knowledge Discovery and Data Mining (pp. 1173-1182).
3. Nakamura, T., Imamura, M., Mercer, R. and Keogh, E., 2020, November. Merlin:
Parameter-free discovery of arbitrary length anomalies in massive time series archives.
In 2020 IEEE international conference on data mining (ICDM) (pp. 1190-1195). IEEE.
4. Law, S.M., 2019. STUMPY: A powerful and scalable Python library for time series data
mining. Journal of Open Source Software, 4(39), p.1504.

### Segmentation

#### 1. Time series segmentation

Mentors: Tony Bagnall ({user}`TonyBagnall`) and TBC

##### Description

The time series segmentation module contains a range of algorithms for segmenting time
series. The goal of this project is to extend the functionality of segmentation in
`aeon` and develop tools for comparing segmentation algorithms.

##### Project stages

1. Learn about `aeon` best practices, coding standards and testing policies.
2. Study the existing segmentation algorithms in `aeon`.
3. Implement existing segmentation algorithms, e.g.
https://github.com/aeon-toolkit/aeon/issues/948
4. Implement tools for comparing segmentation algorithms
5. Conduct a bake off of segmentation algorithms on a range of datasets.

##### Project Outcome

As with all research programming based projects, progress can be hindered by many
unforseen circumstances. Success will be measured by engagement, effort and
willingness to join the community rather than performance of the algorithms.

##### References

1. Allegra, M., Facco, E., Denti, F., Laio, A. and Mira, A., 2020. Data segmentation
based on the local intrinsic dimension. Scientific Reports, 10(1), p.16449.
2. Ermshaus, A., Schäfer, P. and Leser, U., 2023. ClaSP: parameter-free time series
segmentation. Data Mining and Knowledge Discovery, 37(3), pp.1262-1300.
3. Hallac, D., Nystrup, P. and Boyd, S., 2019. Greedy Gaussian segmentation of
multivariate time series. Advances in Data Analysis and Classification, 13(3),
pp.727-751.
4. Matteson, D.S. and James, N.A., 2014. A nonparametric approach for multiple change
point analysis of multivariate data. Journal of the American Statistical Association,
109(505), pp.334-345.
5. Sadri, A., Ren, Y. and Salim, F.D., 2017. Information gain-based metric for
recognizing transitions in human activities. Pervasive and Mobile Computing, 38,
pp.92-109.

### Transformation

#### 1. Improve ROCKET transformers

Mentors: Ali Ismail-Fawaz ({user}`hadifawaz1999`) and Matthew Middlehurst
({user}`MatthewMiddlehurst`)

##### Description

The ROCKET algorithm [1] is a very fast and accurate transformation designed for time
series classification. It is based on a randomly initialised convolutional kernels that
are applied to the time series and used to extract summary statistics. ROCKET has
applications to time series classification, extrinsic regression and anomaly detection,
but as a fast and unsupervised transformation, it has potential to a wide range of
other time series tasks.

`aeon` has implementations of the ROCKET transformation and its variants, including
MiniROCKET [2] and MultiROCKET [3]. However, these implementations have room for
improvement ([#208](https://github.com/aeon-toolkit/aeon/issues/208)). There is scope
to speed up the implementations, and the amount of varients is likely unnecessary and
could be condensed into higher quality estimators.

This projects involves improving the existing ROCKET implementations in `aeon` or
implementing new ROCKET variants. The project will involve benchmarking to ensure that
the new implementations are as fast and accurate as the original ROCKET algorithm and
potentially to compare to other implementations ([#214](https://github.com/aeon-toolkit/aeon/issues/214)).
Besides improving the existing implementations, there is scope to implement the HYDRA
algorithm [4] or implement GPU compatible versions of the algorithms.

##### Project Stages

1. Learn about `aeon` best practices, coding standards and testing policies.
2. Study the ROCKET, MiniROCKET, MultiROCKET algorithms.
3. Study the existing ROCKET implementations in `aeon`.
4. Merge and tidy the ROCKET implementations, with the aim being to familiarise the
mentee with the `aeon` pull request process.
5. Implement one (or more) of the proposed ROCKET implementation improvements:
   * Significantly alter the current ROCKET implementations with the goal of
   speeding up the implementation on CPU processing.
   * Implement a GPU version of some of the ROCKET transformers, using either
   `tensorflow` or `pytorch`.
   * Extend the existing ROCKET implementations to allow for the use of unequal length
   series.
   * Implement the HYDRA algorithm.
6. Benchmark the implementation against the original ROCKET implementations, looking at
booth speed of the transform and accuracy in a classification setting.

##### Project Outcomes

Success of the project will be assessed by the quality of the code produced and an
evaluation of the transformers in a classification setting. None of the implementations
should significantly degrade the performance of the original ROCKET algorithm in terms
of accuracy and speed. Regardless, effort and engagement with the project and the
`aeon` community are more important factors in evaluating success.

##### References

1. Dempster, A., Petitjean, F. and Webb, G.I., 2020. ROCKET: exceptionally fast and
accurate time series classification using random convolutional kernels.
Data Mining and Knowledge Discovery, 34(5), pp.1454-1495.
2. Dempster, A., Schmidt, D.F. and Webb, G.I., 2021, August. Minirocket: A very fast
(almost) deterministic transform for time series classification. In Proceedings of the
27th ACM SIGKDD conference on knowledge discovery & data mining (pp. 248-257).
3. Tan, C.W., Dempster, A., Bergmeir, C. and Webb, G.I., 2022. MultiRocket: multiple
pooling operators and transformations for fast and effective time series classification.
Data Mining and Knowledge Discovery, 36(5), pp.1623-1646.
4. Dempster, A., Schmidt, D.F. and Webb, G.I., 2023. Hydra: Competing convolutional
kernels for fast and accurate time series classification. Data Mining and Knowledge
Discovery, pp.1-27.

### Documentation

#### 1. Improve automated API documentation

Mentors: Matthew Middlehurst ({user}`MatthewMiddlehurst`)

##### Description

`aeon` uses `sphinx` and `numpydoc` to generate API documentation from docstrings.
Many of the docstrings are incomplete or missing sections, and could be improved to
make the API documentation more useful. The goal of this project is to generally
improve the API documentation. A specific goal is to automatically generate links to
examples which use the function/class, similar to the `scikit-learn` documentation.
The way this is achieved is up to the mentee, but should include a new section in the
relevant API page. I.e., the API page for
`aeon.transformers.collection.convolution_based.Rocket` should have a section called
"Examples" which links to the examples which use the class (such as the Rocket
notebook).

##### Project Stages

1. Learn about `aeon` best practices and project documentation.
2. Familiarise with `sphinx` documentation generation and `numpydoc` docstring
standards.
3. Improve the API documentation for a few classes/functions and go through the Pull
Request and review process.
4. Implement a function or improve the API template to automatically generate links
to examples which use the function/class.
5. The main bulk of work is done, but the API documentation is vast and can always be
improved! If time allows, continue to enhance the API documentation through individual
docstrings, API landing page and template improvements at the mentees discretion.

##### Project Outcomes

Success of the project will be assessed by the quality of the documentation produced
and engagement with the project and the `aeon` community. Automatically generating
links to examples is the primary goal.
