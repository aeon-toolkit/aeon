.. _distances_ref:

Distances
=========

The :mod:`aeon.distances` module contains time series specific distance functions
that can be used in aeon and scikit learn estimators. It also contains tools for
extracting the alignment paths for a distance calculation between two series, and
tools for finding all pairwise distances

Amerced Dynamic Time Warping (ADTW)
-----------------------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    adtw_distance
    adtw_pairwise_distance
    adtw_cost_matrix
    adtw_alignment_path

Dynamic Time Warping (DTW)
--------------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    dtw_distance
    dtw_pairwise_distance
    dtw_cost_matrix
    dtw_alignment_path

Edit Real Penalty (ERP)
-----------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    erp_distance
    erp_pairwise_distance
    erp_cost_matrix
    erp_alignment_path

Edit distance for real sequences (EDR)
--------------------------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    edr_distance
    edr_pairwise_distance
    edr_cost_matrix
    edr_alignment_path

Euclidean
---------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    euclidean_distance
    euclidean_pairwise_distance

Longest Common Subsequence (LCSS)
---------------------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    lcss_distance
    lcss_pairwise_distance
    lcss_cost_matrix
    lcss_alignment_path

Manhattan
---------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    manhattan_distance
    manhattan_pairwise_distance

Minkowski
---------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    minkowski_distance
    minkowski_pairwise_distance

Move-Split-Merge (MSM)
----------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    msm_distance
    msm_pairwise_distance
    msm_cost_matrix
    msm_alignment_path

Shape-based Distance (SBD)
--------------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    sbd_distance
    sbd_pairwise_distance

Shape Dynamic Time Warping (Shape DTW)
--------------------------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    shape_dtw_distance
    shape_dtw_pairwise_distance
    shape_dtw_cost_matrix
    shape_dtw_alignment_path

Squared
-------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    squared_distance
    squared_pairwise_distance

Time Warp Edit (TWE)
--------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    twe_distance
    twe_pairwise_distance
    twe_cost_matrix
    twe_alignment_path

Weighted Derivative Dynamic Time Warping (WDDTW)
------------------------------------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    wddtw_distance
    wddtw_pairwise_distance
    wddtw_cost_matrix
    wddtw_alignment_path

Weighted Dynamic Time Warping (DTW)
-----------------------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    wdtw_distance
    wdtw_pairwise_distance
    wdtw_cost_matrix
    wdtw_alignment_path


General methods with distance argument
--------------------------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    distance
    pairwise_distance
    cost_matrix
    alignment_path
    create_bounding_matrix

General methods to recover distance functions
---------------------------------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    get_distance_function
    get_pairwise_distance_function
    get_distance_function_names
    get_cost_matrix_function
    get_alignment_path_function
