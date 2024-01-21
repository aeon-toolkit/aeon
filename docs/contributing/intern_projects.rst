.. _gsoc_projects:

=====================================
Google Summer of code (GSoC) projects
=====================================

Introduction
============
We run a range of short projects with aeon, including undergraduate/postgraduate
project and paid and unpaid internships. We aim to run schemes to help The goal is to
help new contributors to become more familiar with aeon, time series machine
learning, and open-source software development.

aeon projects for 2024
======================
This is a list of some of the projects we are interested in running in 2024. We will
more widelty advertise funding opportunities as and when they become available. Feel
fre to propose your own project ideas, but please discuss them with us first. We have
an active community of researchers and students who work on aeon. Please get in touch
via slack if you are interested in any of these projects or have any questions.

Optimizing the Shapelet Transform
---------------------------------

Mentors : Antoine Guillaume (@baraline)
***************************************

Description
***********
A shapelet is defined as a time series subsequence representing a pattern of interest that we wish to search for in time series data. Shapelet-based algorithm can be used for a wide range of time series task, in this project, we will focus on its core application, which is to create an embedding of the input time series.

Our goal in this project will be to optimize the code related to the shapelet transform method, which takes as input a set of shapelets and a time series dataset, and give as output a tabular dataset containing the features characterizing the presence (or absence) of each shapelet in the input time series (more information in [1]_[2]_).

Similarity search is another field of time series, which has proposed greatly optimized algorithms (see [3]_[4]_) for the task of finding the best matches of a subsequence inside another time series. As this task is extremely similar to what is done in the shapelet transform, we want to adapt these algorithms to the context of shapelets, in order to achieve significant speed-ups.

Project stages
**************
To achieve this goal, with the assistance of the mentor, we identify the following steps for the future GSoC Contributor:
    0. Learn about aeon best practices, coding standards and testing policies.
	1. Study the shapelet transform algorithm and how it is related to the task of similarity search.
	2. Study the similarity search algorithms for the Euclidean distance and the computational optimization they use.
	3. Propose a candidate implementation for to increase the performance of the computations made by a single shapelet. This can be made with the help of the existing implementation of the similarity search module in aeon.
    4. Measure the performance of this first candidate implementation against the current approach.
	5. Implement this solution to the shapelet transform algorithm, which use multiple shapelets.
	6. Benchmark the implementation against the original shapelet transform algorithm.
	7. If time, generalize this new algorithm to the case of dilated shapelets (see [5]_)

Project evaluation
******************
Based on the benchmark of the different implementations, we will evaluate the performance gains of the new shapelet transform and the success of this project.

References
**********
.. [1] Jon Hills et al., "Classification of time series by shapelet transformation",
   Data Mining and Knowledge Discovery, 28(4), 851--881, 2014.
.. [2] A. Bostrom and A. Bagnall, "Binary Shapelet Transform for Multiclass Time
   Series Classification", Transactions on Large-Scale Data and Knowledge Centered
   Systems, 32, 2017.
.. [3] Chin-Chia Michael Yeh et al., "Matrix Profile I: All Pairs Similarity Joins for
   Time Series: A Unifying View that Includes Motifs, Discords and Shapelets",
   IEEE ICDM, 2016
.. [4] Yan Zhu et al., "Matrix Profile II: Exploiting a Novel Algorithm and GPUs to
   break the one Hundred Million Barrier for Time Series Motifs and Joins",
   IEEE ICDM, 2016.
.. [5] Antoine Guillaume et al. "Random Dilated Shapelet Transform: A New Approach
   for Time Series Shapelets", Pattern Recognition and Artificial Intelligence.
   ICPRAI 2022.

Time series segmentation
------------------------

Mentors : Tony Bagnall (@TonyBagnall) and ??
********************************************

The time series segmentation module contains a range of algorithms for segmenting time series.
The goal of this project is to extend the functionality of segmentation in aeon
and develop tools for comparing segmentation algorithms.

Project stages
**************
    0. Learn about aeon best practices, coding standards and testing policies.
    1. Study the existing segmentation algorithms in aeon.
    2. Implement exsiting segmentation algorithms, e.g. https://github.com/aeon-toolkit/aeon/issues/948
    3. Implement tools for comparing segmentation algorithms, e.g.
    4. Conduct a bake off of segmentation algorithms on a range of datasets.

Project evaluation
******************
As with all research programming based projects, progress can be hindered by many
unforseen circumstances. Success will be measured by engagement, effort and
willingness to join the community.

References
**********
[1] Allegra, Michele, et al. "Data segmentation based on the local intrinsic
dimension." Scientific reports 10.1 (2020): 1-12.
[2] Ermshaus, Arik, Sch"afer, Patrick and Leser, Ulf. ClaSP: parameter-free
    time series segmentation. Data Mining and Knowledge Discovery, 37, 2023.
[3]  Hallac, D., Nystrup, P. & Boyd, S.
   "Greedy Gaussian segmentation of multivariate time series.",
    Adv Data Anal Classif 13, 727–751 (2019).
[4]  S. Matteson, David S., and Nicholas A. James. "A nonparametric approach for
    multiple change point analysis of multivariate data." Journal of the American
    Statistical Association 109.505 (2014): 334-345.
[5] Sadri, Amin, Yongli Ren, and Flora D. Salim.
       "Information gain-based metric for recognizing transitions in human activities.",
       Pervasive and Mobile Computing, 38, 92-109, (2017).
