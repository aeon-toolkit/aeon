# Continuous integration

We use continuous integration services on GitHub to automatically check
if new pull requests do not break anything and meet code quality
standards such as a common [coding standards](developer_guide/coding_standards.md).
Before setting up Continuous Integration, be sure that you have set
up your developer environment, and installed a [development version](developer_guide/dev_installation.md)
of aeon.

## Code quality checks

We use [pre-commit](https://pre-commit.com) for code quality checks (a process we also refer to as "linting" checks).

We recommend that you also set this up locally as it will ensure that you never run into code quality errors when you make your first PR!
These checks run automatically before you make a new commit.
To setup, simply navigate to the aeon folder and install our pre-commit configuration:

```{code-block} powershell
pre-commit install
```

`pre-commit` should now automatically run anything you make a commit! Please let us know if you encounter any issues getting this setup.

For a detailed guide on code quality and linting for developers, see [coding_standards](developer_guide/coding_standards.md).

## Unit testing

We use [pytest](https://docs.pytest.org/en/latest/) for unit testing.

To check if your code passes all tests locally, you need to install the development version of `aeon` and all extra dependencies.

1. Install the development version of `aeon` with developer dependencies:

```{code-block} powershell
pip install -e .[dev]
```

   This installs an editable [development version](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs)
   of aeon which will include the changes you make.

   For trouble shooting on different operating systems, please see our detailed
   [installation instructions](installation.md).

2. To run all unit tests, run:

```{code-block} powershell
pytest ./aeon
```

## Test coverage

We use [coverage](https://coverage.readthedocs.io/), the [pytest-cov](https://github.com/pytest-dev/pytest-cov) plugin, and [codecov](https://codecov.io) for test coverage.

## Infrastructure

This section gives an overview of the infrastructure and continuous
integration services we use.

| Platform       | Operation                                                                  | Configuration                                                                                                                                          |
| -------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| GitHub Actions | Build/test/distribute on Linux, MacOS and Windows, run code quality checks | [.github/workflows/](https://github.com/aeon-toolkit/aeon/tree/main/.github/workflows)                                                                 |
| Read the Docs  | Build/deploy documentation                                                 | [.readthedocs.yml](https://github.com/aeon-toolkit/aeon/blob/main/.readthedocs.yml)                                                                    |
| Codecov        | Test coverage                                                              | [.codecov.yml](https://github.com/aeon-toolkit/aeon/blob/main/.codecov.yml), [.coveragerc](https://github.com/aeon-toolkit/aeon/blob/main/.coveragerc) |
