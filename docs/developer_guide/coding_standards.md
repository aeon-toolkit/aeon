# Coding Standards

The `aeon` codebase adheres to a number of coding standards. While these can feel
restrictive at times, they are important for keeping the codebase readable and
maintainable in a collaborative environment.

This page provides an overview of the `aeon` coding standards, with the most important
being:

- The [PEP8](https://www.python.org/dev/peps/pep-0008/) coding guidelines. A good
example can be found [here](https://gist.github.com/nateGeorge/5455d2c57fb33c1ae04706f2dc4fee01)
- Code formatting according to [black](https://black.readthedocs.io/) and [flake8](https://flake8.pycqa.org/en/)
- Documentation formatting using the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html)
style

### Code formatting and linting

Our coding standards are enforced through our CI/CD workflows via [pre-commit](https://pre-commit.com/).
We adhere to the code formatting standards using the following `pre-commit` hooks:

- [black](https://black.readthedocs.io/en/stable/) with default settings
- [flake8](https://flake8.pycqa.org/en/latest/) with a `max_line_length=88` and the
`flake8-bugbear` and `flake8-print` plugins
- [isort](https://isort.readthedocs.io/en/latest/) to sort file imports
- [nbQA](https://github.com/nbQA-dev/nbQA) to lint and format Jupyter notebooks using
the above hooks
- [ruff](https://docs.astral.sh/ruff/)'s [pydocstyle](https://docs.astral.sh/ruff/rules/#pydocstyle-d)
module to enforce the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>)
documentation style
- [pyupgrade](https://github.com/asottile/pyupgrade) to upgrade Python syntax to modern
standards
- Some standard [pre-commit hooks](https://pre-commit.com/hooks.html) for general code
quality

The full `pre-commit` configuration can be found in [.pre-commit-config.yaml](https://github.com/aeon-toolkit/aeon/blob/main/.pre-commit-config.yaml).
Additional configurations for some hooks can be found in the [pyproject.toml](https://github.com/aeon-toolkit/aeon/blob/main/pyproject.toml).

### `aeon` specific code formatting conventions

- Check out our [glossary](glossary.md) for
preferred terminology
- Use underscores to separate words in non-class names i.e.`n_cases` rather than
`n_cases`.
- Exceptionally, capital letters `X`, `Y`, `Z`, are permissible as variable names or
part of variable names such as `X_train` if referring to data sets.
- Use absolute imports for references inside `aeon`.
- Don’t use `import *` in the source code. It is considered harmful by the official
Python recommendations.

### Using `pre-commit`

To set up pre-commit, follow these steps in a Python environment with the `aeon`
`dev` dependencies installed.

Type the below in your Python environment, and in the root of your local repository
clone:

1. If not already done, ensure `aeon` with `dev` dependencies is installed, this
includes `pre-commit`:

```{code-block} powershell
pip install -e .[dev]
```

2. Set up pre-commit:

```{code-block} powershell
pre-commit install
```

Once installed, pre-commit will automatically run all `aeon` code quality checks on
the files you changed whenever you make a new commit.

If you want to exclude some line of code from being checked, you can add a `# noqa`
(no quality assurance) comment at the end of that line. This should only be used
sparingly and with good reason. It is best to limit this to specific checks, i.e.
`# noqa: T201` for `print` statements.
