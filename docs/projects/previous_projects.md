# Previous aeon Projects

A list of projects that have been completed in the past or are currently ongoing.

## 2025

### Implementing and Evaluating Machine Learning Forecasters

Mentors: Matthew Middlehurst ({user}`MatthewMiddlehurst`) and Tony Bagnall
({user}`TonyBagnall`)

Mentee: Tina Jin ({user}`TinaJin0228`)

https://summerofcode.withgoogle.com/organizations/numfocus/projects/details/MPYRSOTi

https://medium.com/@jintina48/list/gsoc25-blog-11a0081fc6e2

GSoC 2025 project

### Deep Learning for Forecasting

Mentors: Tony Bagnall ({user}`TonyBagnall`) and Ali Ismail-Fawaz
({user}`hadifawaz1999`) and Matthew Middlehurst ({user}`MatthewMiddlehurst`)

Mentee: Balgopal Moharana ({user}`lucifer4073`)

https://summerofcode.withgoogle.com/organizations/numfocus/projects/details/arjEn266

https://medium.com/@lucifer4073/gsoc-25-journey-af8e3e0c2621

GSoC 2025 project

## 2024

### Developing Deep Learning Framework and Implementations for Time Series Clustering

Mentors: Ali Ismail-Fawaz ({user}`hadifawaz1999`) and Tony Bagnall
({user}`TonyBagnall`) and Matthew Middlehurst ({user}`MatthewMiddlehurst`)

Mentee: Aadya Chinubhai ({user}`aadya940`)

GSoC 2024 project

https://summerofcode.withgoogle.com/programs/2024/projects/Hvd0DfkD

https://medium.com/@aadyachinubhai

#### Project Summary

Time series clustering involves grouping similar time series data together based on
specific features or patterns. Deep learning algorithms have become increasingly
popular for clustering. However, the aeon's deep clustering module currently lacks
several deep learning-based algorithms. In this project the aim is to implement some of
the top performing and interesting algorithms from a recent comparison of deep learning
for time series clustering and benchmark them. This project includes further developing
the aeon deep learning networks module, making the package publicly documented for user
to explore and well tested to help the maintenance of the deep learning implemented in
the future.

### Implement the Proximity Forest Algorithm for Time Series Classification

Mentors: Matthew Middlehurst ({user}`MatthewMiddlehurst`) and Tony Bagnall
({user}`TonyBagnall`) and Antoine Guillaume ({user}`baraline`)

Mentee: Divya Tiwari ({user}`itsdivya1309`)

https://summerofcode.withgoogle.com/programs/2024/projects/8TYGhJjy

https://medium.com/@Divya2003/

#### Project Summary

This project will implement and benchmark the Proximity Forest Algorithm for Time Series
Classification in aeon. With the ever-increasing data, the applications of time series
classification are also increasing. Hence, we need classification algorithms that are
both efficient and scalable. The Proximity Forest Algorithm is the current
state-of-the-art distance-based classifier that creates an ensemble of decision trees,
where the splits are based on the similarity between time series measured using various
parameterised distance measures. Currently, a version of Proximity Forest which can
match the performance of the original implementation has not been implemented in Python.
This project aims to implement Proximity Forest in aeon for the classification of
univariate time series datasets of equal length and make it accessible for a greater
variety of users. The implementation will be benchmarked on the UCR archive to match
the results of the original Java implementation in terms of run time and accuracy.

### Machine learning from EEG with aeon-neuro

Mentors: Tony Bagnall ({user}`TonyBagnall`) and Matthew Middlehurst
({user}`MatthewMiddlehurst`) and Aiden Rushbrooke ({user}`AidenRushbrooke`)

Mentee: Gabriel Riegner ({user}`griegner`)

GSoC 2024 project

https://summerofcode.withgoogle.com/programs/2024/projects/htrPCGOM

https://gist.github.com/griegner/c414f77d957dea73b84dcd80d580b602

#### Project Summary

Develop aeon-neuro to provide structured tools for machine learning from neural data.
This project will focus on implementing algorithms for EEG classification by building
on the multivariate classification algorithms outlined in Rushbrooke 2023. This paper
demonstrates that existing time series models implemented in aeon can successfully
classify patients from healthy individuals using frequency domain features alone,
eliminating the need for detailed time domain feature selection. In addition to
applying existing machine learning models to EEG datasets, we will further develop
aeon-neuro to be more accessible to the scientific research community by interfacing
it with existing data formatting standards (BIDs) and EEG analysis libraries (MNE).
Alongside these primary outcomes, we will adhere to best practices in research software
development, including writing well-test code, consistent documentation, and user-facing
examples/notebooks.
