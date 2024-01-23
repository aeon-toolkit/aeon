.. _clustering_ref:

Clustering
==========

The :mod:`aeon.clustering` module contains algorithms for time series clustering.

All clusterers in `aeon` can be listed using the `aeon.registry.all_estimators`
utility, using `estimator_types="clusterer"`, optionally filtered by tags.
Valid tags can be listed using `aeon.registry.all_tags`.

Deep learning
-------------

.. currentmodule:: aeon.clustering.deep_learning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseDeepClusterer
    AEFCNClusterer

Clustering Algorithms
---------------------

.. currentmodule:: aeon.clustering

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSeriesKMeans
    TimeSeriesKMedoids
    TimeSeriesKShapes
    TimeSeriesKernelKMeans
    TimeSeriesCLARA
    TimeSeriesCLARANS

Base
----

.. currentmodule:: aeon.clustering.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseClusterer
