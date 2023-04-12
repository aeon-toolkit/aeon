
.. _clustering_ref:

Time series clustering
======================

The :mod:`aeon.clustering` module contains algorithms for time series clustering.

All clusterers in ``aeon``can be listed using the ``aeon.registry.all_estimators`` utility,
using ``estimator_types="clusterer"``, optionally filtered by tags.
Valid tags can be listed using ``aeon.registry.all_tags``.

Clustering models
-----------------

.. currentmodule:: aeon.clustering.k_means

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSeriesKMeans

.. currentmodule:: aeon.clustering.k_medoids

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSeriesKMedoids

.. currentmodule:: aeon.clustering.k_shapes

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSeriesKShapes

.. currentmodule:: aeon.clustering.kernel_k_means

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSeriesKernelKMeans

Base
----

.. currentmodule:: aeon.clustering.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseClusterer

.. currentmodule:: aeon.clustering.partitioning

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TimeSeriesLloyds
