# Implementing Estimators

This page describes how to implement `aeon` compatible estimators, and how to ensure and test compatibility. There are additional steps for estimators that are contributed to `aeon` directly.

## Implementing an `aeon` compatible estimator

The high-level steps to implement `aeon` compatible estimators are as follows:

1. Identify the type of the estimator: [forecaster](/examples/forecasting/forecasting.ipynb), [classifier](/examples/classification/classification.ipynb), etc.
2. Copy the [extension template](https://github.com/aeon-toolkit/aeon/tree/main/extension_templates) for that kind of estimator to its intended location and rename it
3. Fill out the extension template
4. Run the `aeon` test suite and/or the `check_estimator` utility (see [here](https://www.aeon-toolkit.org/en/latest/developer_guide/add_estimators.html#using-the-check-estimator-utility))
5. If the test suite highlights bugs or issues, fix them and return to 4

### `aeon` learning tasks and base classes

`aeon` is structured along modules encompassing specific learning tasks, e.g.,
forecasting, classification, regression or segmentation, with a class structure to
reflect that. See the [base class overview](/examples/base/base_classes.ipynb)
for more on the code structure.

We tag each estimator with a type associated with the relevant base classifier. For
example, the type of an estimator that extends `BaseForecaster` is "forecaster" and
the type of an estimator that solves the time series classification task and
extends `BaseClassifier` is "classifier".

Estimators for a given task are located in the respective module, i.e. classifiers will
be found in `classification`. The estimator types also map onto the different extension
templates found in the [extension_templates](https://github.com/aeon-toolkit/aeon/tree/main/extension_templates) directory of `aeon`.

Base classes contain operations common to all algorithms of that type concerning
checking and conversion of input data, and checking that tags match the data. For each
module, the template gives a step by step guide on how to extend the base classes. For
example, `BaseClassifier` defines the `fit` and `predict` base class methods that
handle data checking and conversion. All classifiers extend `BaseClassifier` and
implement the private methods `_fit` and `_predict` which contain the core logic of
the algorithm.

### `aeon` extension templates

Extension templates are convenient "fill-in" templates for implementers of new
estimators. Classes contain `tags` that describe the algorithm and the type of data
it can handle.

To use the `aeon` extension templates, copy them to the intended location of the
estimator. Inside the extension templates, necessary actions are marked with
`todo`. The typical workflow goes through the extension template by searching for
`todo`, and carrying out the action described next to the `todo`.

Extension templates typically have the following `todo`:

- choosing name and parameters for the estimator
- filling in the `__init__`: writing parameters to `self`, calling `super`'s `__init__`
- filling in docstrings of the module and the estimator. This is recommended as early
as parameters have been settled on, it tends to be useful as a specification to follow
in implementation
- filling in the tags for the estimator. Some tags are "capabilities", i.e., what the
estimator can do, e.g., dealing with NaN values. Other tags determine the format of
inputs seen in the "inner" methods `_fit`, etc, these tags are usually called
`X_inner_type` or similar
- Filling in the inherited abstract methods, e.g., `_fit` and `_predict`. The
docstrings and comments in the extension template should be followed here. The
docstrings also describe the guarantees on the inputs to the "inner" methods, which
are typically stronger than the guarantees on inputs to the public methods, and
determined by values of tags that have been set. For instance, setting the tag
`y_inner_type` to `pd.DataFrame` for a forecaster guarantees that the `y` seen by
`_fit` will be a `pandas.DataFrame`, complying with additional data container
specifications in `aeon` (e.g., index types)
- filling in testing parameters in `get_test_params`. The selection of parameters
should cover major estimator internal case distinctions to achieve good coverage

Some common caveats, also described in extension template text:

- `__init__` parameters should be written to `self` and never be changed
  - special case of this: estimator components, i.e., parameters that are estimators,
  should generally be cloned (i.e. via `sklearn.clone`), and method should be called
  only on the clones
- methods should generally avoid side effects on arguments
- non-state changing methods (i.e. `predict` and `transform` should not write to
`self` in general
- typically, implementing `get_params` and `set_params` is not needed, since `aeon`'s
`BaseEstimator` inherits from `sklearn`'s. Custom `get_params`, `set_params` are
typically needed only for complex cases only heterogeneous composites, e.g., pipelines
with parameters that are nested structures containing estimators

### Using the `check_estimator` utility

Usually, the simplest way to test complaince with `aeon` is via the `check_estimator`
methods in the `utils.estimator_checks` module.

When invoked, this will collect tests in `aeon` relevant for the estimator type and
run them on the estimator.

This can be used for manual debugging in a notebook environment. Example of running the
full test suite for `NaiveForecaster`:

```{code-block} powershell
from aeon.testing.estimator_checks import check_estimator
from aeon.forecasting.naive import NaiveForecaster
check_estimator(NaiveForecaster)
```

The `check_estimator` utility will return, by default, a `dict`, indexed by
test/fixture combination strings, that is, a test name and the fixture combination
string in squared brackets. Example: `'test_repr[NaiveForecaster-2]'`, where
`test_repr` is the test name, and `NaiveForecaster-2` the fixture combination string.

Values of the return `dict` are either the string `"PASSED"`, if the test succeeds,
or the exception that the test would raise at failure. `check_estimator` does not raise
exceptions by default, the default is returning them as dictionary values. To raise the
exceptions instead, e.g., for debugging, use the argument `raise_exceptions=True`, which
will raise the exceptions instead of returning them as dictionary values. In that case,
there will be at most one exception raised, namely the first exception encountered in
the test execution order.

To run or exclude certain tests, use the `tests_to_run` or `tests_to_exclude` arguments.
Values provided should be names of tests (str), or a list of names of tests. Note that
test names exclude the part in squared brackets.

Example, running the test `test_constructor` with all fixtures:

```{code-block} powershell
check_estimator(NaiveForecaster, tests_to_run="test_constructor")
```
outputs
`{'test_constructor[NaiveForecaster]': 'PASSED'}`

To run or exclude certain test-fixture-combinations, use the `fixtures_to_run` or
`fixtures_to_exclude` arguments. Values provided should be names of
test-fixture-combination strings (str), or a list of such. Valid strings are precisely
the dictionary keys when using `check_estimator` with default parameters.

Example, running the test-fixture-combination `"test_repr[NaiveForecaster-2]"`:

```{code-block} powershell
check_estimator(NaiveForecaster, fixtures_to_run="test_repr[NaiveForecaster-2]")
```
outputs `{'test_repr[NaiveForecaster-2]': 'PASSED'}`

A useful workflow for using `check_estimator` to debug an estimator is as follows:

1. Run `check_estimator(MyEstimator)` to find failing tests
2. Subset to failing tests or fixtures using `fixtures_to_run` or `tests_to_run`
3. If the failure is not obvious, set `raise_exceptions=True` to raise the exception and inspecet the traceback
4. If the failure is still not clear, use advanced debuggers on the line of code with `check_estimator`

### Running the test suite in a repository clone

If the target location of the estimator is within `aeon`, then the `aeon` test suite
can be run instead. The `aeon` test suite (and CI/CD) is `pytest` based, `pytest` will
automatically collect all estimators of a certain type and tests applying for a given
estimator.

Generic interface compliance tests are contained in the classes `TestAllEstimators`,
`TestAllForecasters`, and so on. `pytest` test-fixture-strings for an estimator
`EstimatorName` will always contain `EstimatorName` as a substring, and are identical
with the test-fixture-strings returned by `check_estimator`.

To run tests only for a given estimator from the console, the command
`pytest -k "EstimatorName"` can be used. This will typically have the same effect as
using `check_estimator(EstimatorName)`, only via direct `pytest` call. When using
Visual Studio Code or pycharm, tests can also be sub-setted using GUI filter
functionality - for this, refer to the respective IDE documentation on test integration.

To identify codebase locations of tests applying to a specific estimator, a quick
approach is searching the codebase for test strings produced by `check_estimator`,
preceded by `def` (for function/method definition).

## Adding an estimator to `aeon`

When adding an `aeon` compatible estimator to `aeon` itself, a number of
additional things need to be done:

- Ensure that code also meets `aeon's` developer documentation standards
- Add the estimator to the `aeon` API reference. This is done by adding a reference to
the estimator in the correct `rst` file inside `docs/api_reference`
- Authors of the estimator should add themselves to `CODEOWNERS`, as owners of the
contributed estimator
- If the estimator relies on soft dependencies, or adds new soft dependencies, the
steps in the [dependencies developer guide](developer_guide/dependencies.md)
should be followed
- Ensure that the estimator passes the entire local test suite of `aeon`, with the
estimator in its target location. To run tests only for the estimator, the command
`pytest -k "EstimatorName"` can be used (or vs code GUI filter functionality)
- Ensure that test parameters in `get_test_params` are chosen such that runtime of
estimator specific tests remains in the seconds order on `aeon` remote CI/CD

When contributing to `aeon`, core developers will give helpful pointers on the above in
their PR reviews. It is recommended to open a draft PR and ask developers for their
opinions to get feedback early.
