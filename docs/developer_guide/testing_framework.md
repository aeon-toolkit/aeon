# Testing framework

`aeon` uses `pytest` for testing interface compliance of estimators, and correctness of code. This page gives an overview over the tests, and introductions on how to add tests, or how to extend the testing framework.

## Test module architecture

`aeon` testing happens on three layers, roughly corresponding to the inheritance layers of estimators.

* **package level**: testing interface compliance with the `BaseObject` and `BaseEstimator` specifications, in `tests/test_all_estimators.py`

* **module level**: testing interface compliance of concrete estimators with their base class, for instance `classification/tests/test_all_classifiers.py`

* **low level**: testing individual functionality of estimators or other code, in individual files in `tests` folders.

The `aeon` testing framework is under redesign. For information on the current state of the testing framework, please
refer to the dcostring documentation of the `testing` module.
