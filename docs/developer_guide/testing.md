# Testing framework

`aeon` uses `pytest` for testing interface compliance of estimators and correctness of
code. This page gives an overview of the test framework.

Unit tests should cover as much code as possible. This includes differing parameters
for functions and estimators, error handling, and edge cases. It is not enough to just
run the code, but to test that it behaves as expected through output and state checks
after.

## Writing `aeon` tests

There are two main ways to test code in `aeon`. Through test files and general
estimator testing.

### Test files

All files, functions, and classes can have corresponding test files in a corresponding
`tests` directory of the package. If `module.py` is the file to be tested, the test
file for it should be placed in `tests/test_module.py` in the same package as a
original file.

```
aeon/
   └── package/
      ├── __init__.py
      ├── module.py
      └── tests/
         ├── __init__.py
         └── test_module.py
```

All unit tests should be placed in a `tests` directory with a filename starting with
`test_`. All test functions should start with `test_` to be discovered by `pytest` i.e.

```python
def test_function():
    assert function() == expected_output
```

For estimators, testing of base class functionality should be avoided to prevent
duplication with the general testing. Avoid testing:
- Basic runs of `fit` and `predict` methods using simple/testing parameters
- Base class functionality such as the output of `get_params` and `set_params` or
converting input data to the correct format.
- Error handling for basic errors such as wrong input types or output shapes and types

Do test:
- Specific functionality of the estimator such as parameters not seen in general testing
- Whether internal attributes are set correctly after fitting or output is as expected
- Edge cases and error handling for parameter values and more complex errors

Test functions which require soft dependencies should be skipped if the dependencies
are not installed. This can be done using the `pytest.mark.skipif` decorator. See the
[dependencies page](#developer_guide/dependencies).

### General estimator testing

The [`testing` module](https://github.com/aeon-toolkit/aeon/tree/main/aeon/testing/)
contains generalised testing for any class which extends
[BaseAeonEstimator](base.BaseAeonEstimator). This will test the estimator against
a set of general checks to ensure it is compliant with the `aeon` API. This includes
checking that inherited methods perform as expected, such as that estimator can be
fitted on dummy data without issue. We will also perform a variety of checks on the
estimator i.e. for picking, whether it is non-deterministic and the state of parameters,
inputs and attributes after method calls.

Estimators which inherit from other base classes i.e. `BaseClassifier` or
`BaseAnomalyDetector` will have additional tests run on them to ensure they
comply with the API of the learning task. The tags of the estimator will also impact
what is tested, such as `X_input_type` affecting the test data used and `cant_pickle`
skip tests which require pickling the estimator.

There is no list of all tests that are run on an estimator, but the code for all checks
can be found in the [`estimator_checks` subpackage](https://github.com/aeon-toolkit/aeon/tree/main/aeon/testing/estimator_checking).
The general tests can be run using functions found in the [`testing` API page](#api_reference/testing).,
with the main function being `check_estimator`. This function will collect all
applicable tests from the various `_yield_*_checks.py` files and run them on the
estimator.

Estimators which require soft dependencies will have all but a few tests checking
behaviour surrounding the soft dependency skipped if the dependency is not installed.

#### Adding new checks to the general testing

To add a new check to the general testing, you must place it in the correct file.
For example, if you want your check to be run on all estimators, it should be in the
`_yield_estimator_checks.py` file. Tests relating to soft dependency checking are in
`_yield_soft_depencency_checks.py`. Tests for all classification estimators are in
`_yield_classification_checks.py` and so on.

There are multiple types of checks which can be added to the general testing, any
new check should be one of the following:
- Check for the estimator class
- Check for the estimator class where a datatype is required
- Check for the estimator instances
- Check for the estimator instances where a datatype is required
- Check for the estimator instances where all acceptable datatypes should be tested

In most cases (including the `aeon` GitHub CI) the general testing will be run on all
parameter sets returned by the `_create_test_instance` method of the estimator.
Which can contain multiple objects with different parameter sets. Each estimator
will have a variety of datatypes for testing data available, this includes types such
as 3D numpy arrays and lists, as well as capabilities such as multivariate and unequal
length data. Running a check multiple times for each datatype can be expensive, so
decide whether it is necessary or if any datatype will suffice.

After writing a new check, it then must be added to the `_yield_*_checks` function
at the top of the file. Place the check in the correct section of the function for
the type of check it is. `yield` the check using `partial`, making sure to include
any parameters for the check so that the output check does not require any input. i.e.

```python
yield partial(
    check_non_state_changing_method,  # check function
    estimator=estimator,  # estimator i from estimator_instances
    datatype=datatypes[i][0],  # estimator i, test datatype 0
)
```
See the [`_yield_classification_checks`](https://github.com/aeon-toolkit/aeon/blob/main/aeon/testing/estimator_checking/_yield_classification_checks.py)
function for an example of this.

#### Adding a new module to the general testing

If you have a new module which requires general testing, you must add a new file
to the `estimator_checking` directory. This file should be called `_yield_*_checks.py`
where `*` is the name of the module. This file should follow the same structure as
the other files in the directory.

Ensure the base class for the new module is in the [register](https://github.com/aeon-toolkit/aeon/blob/main/aeon/utils/base/_register.py),
and that it is included as a valid option in any relevant [tags](https://github.com/aeon-toolkit/aeon/blob/main/aeon/utils/tags/_tags.py).

You will have to make sure that valid testing data and labels are available for your
estimator type in [`testing/testing_data.py`](https://github.com/aeon-toolkit/aeon/blob/main/aeon/testing/testing_data.py).
Add to the `FULL_TEST_DATA_DICT` dictionary and edit the functions at the
bottom as necessary to accommodate your module.

Some testing utilities may also need to be edited depending on the module structure,
i.e. [`_run_estimator_method`](https://github.com/aeon-toolkit/aeon/blob/main/aeon/testing/utils/estimator_checks.py)
may need to be edited to accommodate different method parameter names.

#### Excluding tests and estimators

Tests and estimators can be completely excluded from the general testing by adding them
to the `EXCLUDED_ESTIMATORS` and `EXCLUDED_TESTS` lists in the
[`testing/testing_config.py`](https://github.com/aeon-toolkit/aeon/blob/main/aeon/testing/testing_config.py)
file. `EXCLUDED_ESTIMATORS` only requires the estimator class name to skip all tests,
while `EXCLUDED_TESTS` requires the class name and test names in a list.

These skips are only intended as a temporary measure, the issue causing the skip should
be fixed eventually and the item removed. If the issue cannot be resolved in the
estimators themselves, (i.e. `predict` must update the state for the task, which
is not normally allowed) the testing itself should be updated and the items removed
from the exclusion lists.

## Testing in `aeon` CI

`aeon` uses GitHub Actions for continuous integration testing.

The `aeon` periodic test workflow runs once every day (except for manual runs) and will
run all possible tests. This includes running all test files and general estimator
testing on all estimators. The CI will also check for code formatting and linting,
as well as test coverage.

The `aeon` PR testing workflow runs on every PR to the main branch. By default, this
will run a constrained set of tests excluding some tests such as those which
are noticeably expensive or prone to failure (i.e. I/O from external sources).
The estimators run will also be split into smaller subsets to spread them over
different Python version and operating system combinations. This is controlled by the
`PR_TESTING` flag in [`testing/testing_config.py`](https://github.com/aeon-toolkit/aeon/blob/main/aeon/testing/testing_config.py).

A large portion of testing time is spent compiling `numba` functions. By default,
pull request workflows will use a cached set of functions generated from the
periodic test. The cached functions for any changed files will be invalidated.

There are a number of labels which can be added to a PR to control the testing. These
are:
- `codecov actions` - Run the codecov action to update the test coverage
- `full examples run` - Run the full examples in the documentation
- `full pre-commit` - Run the pre-commit checks on all files
- `full pytest actions` - Run all tests in the CI, disable PR_TESTING
- `no numba cache` - Disable the GitHub `numba` cache for tests
- `run typecheck test` - Run the `mypy` typecheck workflow

The periodic tests will run all of the above.

## Running unit tests locally using `pytest`


To check if your code passes all tests locally, you need to install the development
version of `aeon` and all extra dependencies. See the [developer installation guide](#developer_guide/dev_installation)
for more information.

To run all unit tests, run:

```{code-block} powershell
pytest aeon/
```

All regular `pytest` configuration is applicable here. See their [documentation](https://docs.pytest.org/en/stable/index.html)
for more information.

The `-k`  option can be used to run tests with a specific keyword i.e. to run all tests
containing `DummyClassifier`:

```{code-block} powershell
pytest aeon/ -k DummyClassifier
```

All general tests will contain the estimator name in the test name, so this can be
used to run tests for a specific estimator. This will also work for test/check names,
or a combination to run a specific check on a specific estimator.

The `pytest-xdist` dependency allows for parallel testing. To run tests in parallel
on all available cores, run:

```{code-block} powershell
pytest aeon/ -n auto
```

Alternatively, input a number to run on that many cores i.e. `-n 4` to run on 4 cores.

`aeon` also has some custom configuration options in its [conftest.py` file](https://github.com/aeon-toolkit/aeon/blob/main/conftest.py).
There are:
- `--nonumba` - Disable `numba` compilation if true
- `--enablethreading` - Skip setting various threading options to 1 prior to tests if true
- `--prtesting` - Set the PR_TESTING flag

## Tracking test coverage

We use [coverage](https://coverage.readthedocs.io/), the [pytest-cov](https://github.com/pytest-dev/pytest-cov)
plugin, and [codecov](https://codecov.io) for test coverage. Tes coverage can be found
on the [`aeon` codecov page](https://app.codecov.io/gh/aeon-toolkit/aeon).

Workflows which generate coverage reports will have `numba` `njit` functions disabled.
This is mainly because the coverage of these functions cannot be accurately measured.
`numba` functions are also prone to accidental errors such as out-of-bounds array
access, which will not raise an error. As such, we use these workflows as an additional
check for bugs in the codebase.
