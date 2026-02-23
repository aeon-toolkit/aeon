.. _similarity_search_ref:

Similarity search
=================

The :mod:`aeon.similarity_search` module contains algorithms and tools for similarity
search tasks. First, we distinguish between `series` estimator and `collection`
estimators, similarly to the `aeon.transformer` module. Secondly, we distinguish between
estimators used `neighbors` (with suffix SNN for subsequence nearest neighbors, or ANN
for approximate nearest neighbors) search and estimators used for `motifs` search.


Series Similarity search estimators
-----------------------------------

.. currentmodule:: aeon.similarity_search.series.neighbors

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DummySNN
    MassSNN

.. currentmodule:: aeon.similarity_search.series.motifs

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    StompMotif


Collection Similarity search estimators
-----------------------------------

.. currentmodule:: aeon.similarity_search.collection.neighbors

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RandomProjectionIndexANN


Base Estimators
---------------

.. currentmodule:: aeon.similarity_search._base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseSimilaritySearch


.. currentmodule:: aeon.similarity_search.series._base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseSeriesSimilaritySearch


.. currentmodule:: aeon.similarity_search.collection._base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseCollectionSimilaritySearch
