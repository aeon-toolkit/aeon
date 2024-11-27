.. _similarity_search_ref:

Similarity search
=================

The :mod:`aeon.similarity_search` module contains algorithms and tools for similarity
search tasks.


Similarity search estimators
----------------------------

.. currentmodule:: aeon.similarity_search

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    QuerySearch
    SeriesSearch

Distance profile functions
--------------------------

.. currentmodule:: aeon.similarity_search.distance_profiles

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    euclidean_distance_profile
    normalised_euclidean_distance_profile
    squared_distance_profile
    normalised_squared_distance_profile

Matrix profile functions
--------------------------

.. currentmodule:: aeon.similarity_search.matrix_profiles

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    stomp_normalised_euclidean_matrix_profile
    stomp_euclidean_matrix_profile
    stomp_normalised_squared_matrix_profile
    stomp_squared_matrix_profile

Base
----

.. currentmodule:: aeon.similarity_search.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseSimilaritySearch
