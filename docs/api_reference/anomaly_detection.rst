.. _anomaly_detection_ref:

Anomaly Detection
=================

The :mod:`aeon.anomaly_detection` module contains algorithms and composition tools for
time series anomaly detection.

All detectors in `aeon`  can be listed using the `aeon.utils.discovery.all_estimators` utility,
using ``estimator_types="anomaly-detector"``, optionally filtered by tags.
Valid tags can be listed by calling the function `aeon.utils.tags.all_tags_for_estimator`.

Each detector in this module specifies its supported input data format, output data
format, and learning type as an overview table in its documentation. Some detectors
support multiple learning types.


Collection anomaly detectors
----------------------------

.. currentmodule:: aeon.anomaly_detection.collection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClassificationAdapter
    OutlierDetectionAdapter
    BaseCollectionAnomalyDetector

Series anomaly detectors
------------------------

Distance-based
~~~~~~~~~~~~~~

.. currentmodule:: aeon.anomaly_detection.series.distance_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CBLOF
    KMeansAD
    LeftSTAMPi
    LOF
    MERLIN
    STOMP
    ROCKAD

Distribution-based
~~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.anomaly_detection.series.distribution_based

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    COPOD
    DWT_MLEAD

Outlier-Detection
~~~~~~~~~~~~~~~~~

.. currentmodule:: aeon.anomaly_detection.series.outlier_detection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    IsolationForest
    OneClassSVM
    STRAY

Adapters
~~~~~~~~

.. currentmodule:: aeon.anomaly_detection.series

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PyODAdapter

Base
~~~~

.. currentmodule:: aeon.anomaly_detection.series

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseSeriesAnomalyDetector


Base
----

.. currentmodule:: aeon.anomaly_detection.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseAnomalyDetector
