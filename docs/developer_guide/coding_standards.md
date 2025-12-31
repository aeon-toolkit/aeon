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

## Code formatting and linting

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

## `aeon` specific code formatting conventions

- Use underscores to separate words in non-class names i.e.`n_cases` rather than
`ncases`,  `nCases` or similar.
- Exceptionally, capital letters i.e. `X` are permissible as variable names or
part of variable names such as `X_train` if referring to data sets.
- Use absolute imports for references inside `aeon`.
- Don't use `import *` in the source code. It is considered harmful by the official
Python recommendations.

## Referencing externally written code

When using code from another package, adapting code from external sources, or writing
code inspired by another implementation, proper attribution and license information must
be included. This is both a legal requirement and an ethical practice in open-source
development.

### When attribution is required

Attribution is required when:

- You copy or adapt code from another open-source project
- You use code inspired by another implementation
- You port an algorithm from another language or package
- You use utility functions or helper code from external sources

Even if you significantly modify the code, if it was originally based on external
source material, attribution should be provided.

### License compatibility

Before using code from another package, ensure that the license is compatible with
`aeon`'s BSD-3-Clause license. Common compatible licenses include:

- BSD-2-Clause, BSD-3-Clause
- MIT
- Apache 2.0
- Public domain

If the source code uses a different license (e.g., GPL, AGPL), using the code as-is
may not be acceptable. Consult with core developers if you're unsure about license
compatibility.

### Attribution format

When attributing external code, include the following information in an appropriate
docstring (module-level or function/class-level):


1. **A clear statement** that the code was adapted/inspired from external sources
2. **Links to the original code** (preferably to specific files or functions)
3. **Copyright information** including the copyright holder and year
4. **License information** (e.g., BSD-2, BSD-3, MIT)

The attribution should be placed in the module-level docstring or the function/class
docstring, typically in a **Notes** section.

### Example attribution format

Here is the recommended format for attributing external code:

```python
"""Itakura parallelogram bounding matrix.

This code was adapted from pyts and tslearn.

- pyts code:
  https://pyts.readthedocs.io/en/latest/_modules/pyts/metrics/dtw.html#itakura_parallelogram
  Copyright (c) 2018, Johann Faouzi and pyts contributors, BSD-3

- tslearn code (line 974):
  https://github.com/tslearn-team/tslearn/blob/main/tslearn/metrics/dtw_variants.py
  Copyright (c) 2017, Romain Tavenard, BSD-2
"""
```

For module-level attribution, you can use this format:

```python
"""
Greedy Gaussian Segmentation (_GGS).

[... description of the module ...]

Notes
-----
Based on the work from [1]_.

- source code adapted based on: https://github.com/cvxgrp/GGS
  Copyright (c) 2018, Stanford University Convex Optimization Group, BSD-2
- paper available at: https://stanford.edu/~boyd/papers/pdf/ggs.pdf

References
----------
.. [1] Hallac, D., Nystrup, P. & Boyd, S.
   "Greedy Gaussian segmentation of multivariate time series.",
    Adv Data Anal Classif 13, 727–751 (2019).
    https://doi.org/10.1007/s11634-018-0335-0
"""
```

### What to include in your PR

When submitting a PR that includes externally sourced code:

1. **Mention the source** in your PR description
2. **Include proper attribution** in the code docstrings (as shown above)
3. **Verify license compatibility** and mention it in the PR
4. **Link to the original source** in your PR description

Attribution must appear in the source code itself; attribution only in
commit messages or PR descriptions is not sufficient.

Failure to properly attribute external code may result in your PR being closed.

### Additional resources

- For more information on open-source licenses, see the [Open Source Initiative](https://opensource.org/licenses)
- For guidance on contributing, see the [contributing guide](../contributing.md)
- For documentation standards, see the [documentation guide](documentation.md)

## Using `pre-commit`

To set up pre-commit, follow these steps in a Python environment with the `aeon`
`dev` dependencies installed.

Type the below in your Python environment, and in the root of your local repository
clone:

1. If not already done, ensure `aeon` with `dev` dependencies is installed, this
includes `pre-commit`:

```{code-block} powershell
pip install --editable .[dev]
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
