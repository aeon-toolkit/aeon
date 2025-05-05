.. _anomaly_detection_ref:

Anomaly Detection
=================

The :mod:`aeon.anomaly_detection` module contains algorithms and composition tools for time series classification.

All detectors in `aeon`  can be listed using the `aeon.utils.discovery.all_estimators` utility,
using ``estimator_types="anomaly-detector"``, optionally filtered by tags.
Valid tags can be listed by calling the function `aeon.utils.discovery.all_tags_for_estimator`.

Each detector in this module specifies its supported input data format, output data
format, and learning type as an overview table in its documentation. Some detectors
support multiple learning types.

.. note::

    Not all algorithm families are currently implemented. The documentation includes
    placeholders for planned categories which will be supported in future.

Distance-based
--------------

.. currentmodule:: aeon.anomaly_detection.distance_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CBLOF
    KMeansAD
    LeftSTAMPi
    LOF
    MERLIN
    OneClassSVM
    STOMP
    ROCKAD

Distribution-based
-----------------

.. currentmodule:: aeon.anomaly_detection.distribution_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    COPOD
    DWT_MLEAD

Encoding-based
--------------

The algorithms for this family are not implemented yet.

Forecasting-based
-----------------

The algorithms for this family are not implemented yet.

Outlier-Detection
-----------------

.. currentmodule:: aeon.anomaly_detection.outlier_detection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    IsolationForest
    PyODAdapter
    STRAY

Reconstruction-based
--------------------

The algorithms for this family are not implemented yet.


Whole-series
------------

.. currentmodule:: aeon.anomaly_detection.whole_series

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseCollectionAnomalyDetector
    ClassificationAdapter
    OutlierDetectionAdapter

Base
----

.. currentmodule:: aeon.anomaly_detection.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseAnomalyDetector
