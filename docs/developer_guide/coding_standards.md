# Coding standards

## Coding style

We follow:

* the [PEP8](https://www.python.org/dev/peps/pep-0008/) coding guidelines. A good example can be found [here](https://gist.github.com/nateGeorge/5455d2c57fb33c1ae04706f2dc4fee01)

* code formatting according to `black`, `flake8`, `isort`, `numpydoc`

### Code formatting and linting

We adhere to the following code formatting standards:

* [`black`](https://black.readthedocs.io/en/stable/) with default settings

* [`flake8`](https://flake8.pycqa.org/en/latest/) with a `max_line_length=88` and
  some exceptions as per `setup.cfg`

* [`isort`](https://isort.readthedocs.io/en/latest/) with default settings

* [`numpydoc`](https://numpydoc.readthedocs.io/en/latest/format.html) to enforce numpy [docstring standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>) , along with aeon specific conventions described in our [`developer_guide`'s](https://github.com/aeon-toolkit/aeon/tree/main/docs/developer_guide).

This is enforced through our CI/CD workflows via [`pre-commit`](https://pre-commit.com/)

The full pre-commit configuration can be found in [`.pre-commit-config.yaml`](https://github.com/aeon-toolkit/aeon/blob/main/.pre-commit-config.yaml). Additional configurations can be found in [`pyproject.toml`](https://github.com/aeon-toolkit/aeon/blob/main/pyproject.toml).

### `aeon` specific code formatting conventions

- Check out our [`glossary`](https://www.aeon-toolkit.org/en/stable/glossary.html).
- Use underscores to separate words in non-class names: `n_instances` rather than `ninstances`.
- exceptionally, capital letters `X`, `Y`, `Z`, are permissible as variable names or part of variable names such as `X_train` if referring to data sets, in accordance with the PEP8 convention that such variable names are permissible if in prior use in an area (here, this is the `scikit-learn` adjacenet ecosystem)
- Avoid multiple statements on one line. Prefer a line return after a control flow statement (`if`/`for`).
- Use absolute imports for references inside aeon.
- Don’t use `import *` in the source code. It is considered harmful by the official Python recommendations. It makes the code harder to read as the origin of symbols is no longer explicitly referenced, but most important, it prevents using a static analysis tool like pyflakes to automatically find bugs.

### Setting up local code quality checks

There are two options to set up local code quality checks:

* using `pre-commit` for automated code formatting
* setting up `black`, `flake8`, `isort` and/or `numpydoc` manually in a local dev IDE

#### Using pre-commit

To set up pre-commit, follow these steps in a python environment with the `aeon` `dev` dependencies installed.

Type the below in your python environment, and in the root of your local repository clone:

1. If not already done, ensure `aeon` with `dev` dependencies is installed, this includes `pre-commit`:

```{code-block} powershell
pip install -e .[dev]
```
2. Set up pre-commit:

```{code-block} powershell
pre-commit install
```
Once installed, pre-commit will automatically run all `aeon` code quality checks on the files you changed whenever you make a new commit.

If you want to exclude some line of code from being checked, you can add a `#noqa` (no quality assurance) comment at the end of that line.

#### Integrating with your local developer IDE

Local developer IDEs will usually integrate with common code quality checks, but need setting them up in IDE specific ways.

For Visual Studio Code, `black`, `flake8`, `isort` and/or `numpydoc` will need to be activated individually in the preferences (e.g., search for `black` and check the box). The packages `black` etc will need to be installed in the python environment used by the IDE, this can be achieved by an install of `aeon` with `dev` dependencies.

Visual Studio Code preferences also allow setting of parameters such as `max_line_length=88` for `flake8`.

In Visual Studio Code, we also recommend to add `"editor.ruler": 88` to your local `settings.json` to display the max line length.
