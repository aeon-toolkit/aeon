# Installation

`aeon` currently supports Python versions 3.11, 3.12, 3.13 and 3.14. It is available for
most well-known operating systems, and is frequently tested on macOS, Ubuntu and
Windows servers by our development CI.

There are three ways to install a release of `aeon`. Pick one in the tabs of the first
step, the rest of the page follows your choice.

- `pip` installs from [PyPI](https://pypi.org/project/aeon/). This is the recommended
  option for most users.
- [`uv`](https://docs.astral.sh/uv/) also installs from PyPI. It is a fast replacement
  for `pip` which can install Python for you.
- `conda` installs from [conda-forge](https://anaconda.org/conda-forge/aeon).

You can also [install the latest development version](#install-the-latest-development-version)
from GitHub. Building the package from source is a requirement for users who wish to
develop the `aeon` codebase and for most other contributions to the project, see our
[developer installation guide](developer_guide/dev_installation.md).

:::{tip}
To try `aeon` without installing anything, `uv` can run Python in a temporary
environment:

```{code-block} bash
uv run --with aeon python
```
:::

```{note}
While we try to keep output similar between OS and Python version, we cannot
guarantee estimators will output the same results for macOS ARM processors.
```

(using-a-pip-venv)=
## Step 1 - Create an environment

In order to avoid potential conflicts with other packages, we strongly recommend
installing `aeon` in a virtual environment or a fresh conda environment.

::::::{tab-set}
:sync-group: tool

:::::{tab-item} pip
:sync: pip

Prior to this, please ensure you have a compatible version of Python installed
(i.e. from [python.org](https://www.python.org)). Then create a
[virtual environment (venv)](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
and activate it. The name `aeon-venv` can be replaced with a name of your choosing.

::::{tab-set}
:sync-group: os

:::{tab-item} Windows
:sync: windows

```{code-block} powershell
python -m venv aeon-venv
aeon-venv\Scripts\activate
```
:::

:::{tab-item} macOS and Linux
:sync: unix

```{code-block} bash
python3 -m venv aeon-venv
source aeon-venv/bin/activate
```
:::

::::
:::::

:::::{tab-item} uv
:sync: uv

[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/), then create
a virtual environment and activate it. `uv` downloads the requested version of Python
if you do not have it. The environment is created in the `.venv` folder.

::::{tab-set}
:sync-group: os

:::{tab-item} Windows
:sync: windows

```{code-block} powershell
uv venv --python 3.13
.venv\Scripts\activate
```
:::

:::{tab-item} macOS and Linux
:sync: unix

```{code-block} bash
uv venv --python 3.13
source .venv/bin/activate
```
:::

::::
:::::

:::::{tab-item} conda
:sync: conda

Create a new environment and activate it. The name `aeon-env` can be replaced with a
name of your choosing.

```{code-block} bash
conda create -n aeon-env python=3.13
conda activate aeon-env
```
:::::

::::::

Note that this will only activate the environment for the current terminal session.
If you wish to use the environment in a different terminal session, you will need to
activate it again.

(install-the-latest-release-from-pypi)=
(install-the-latest-release-from-conda-forge)=
## Step 2 - Install aeon

All installation options include the core dependencies required to run the framework
components of `aeon`. Some estimators need [optional dependencies](#optional-dependencies),
which the `all_extras` modifier installs with `aeon`.

:::::{tab-set}
:sync-group: tool

::::{tab-item} pip
:sync: pip

To install the latest `aeon` release with core dependencies:

```{code-block} bash
pip install -U aeon
```

To install `aeon` with all stable dependencies. This will also install core
dependencies, so the above command is not required.

```{code-block} bash
pip install -U "aeon[all_extras]"
```
::::

::::{tab-item} uv
:sync: uv

To install the latest `aeon` release with core dependencies:

```{code-block} bash
uv pip install -U aeon
```

To install `aeon` with all stable dependencies. This will also install core
dependencies, so the above command is not required.

```{code-block} bash
uv pip install -U "aeon[all_extras]"
```

:::{dropdown} Using a uv project instead
If you manage your own project with `uv`, add `aeon` to its dependencies. This
records it in your `pyproject.toml` and creates the environment for you, so you can
skip the first step. `uv run` uses this environment without activating it.

```{code-block} bash
uv init
uv add aeon  # or "aeon[all_extras]"
uv run python my_script.py
```
:::
::::

::::{tab-item} conda
:sync: conda

To install the latest `aeon` release in the active environment:

```{code-block} bash
conda install -c conda-forge aeon
```

Currently for `conda` installations, optional dependencies must be installed
separately.
::::

:::::

## Step 3 - Check the installation

This prints the installed version of `aeon`, whichever way you installed it:

```{code-block} bash
python -c "import aeon; print(aeon.__version__)"
```

:::::{tab-set}
:sync-group: tool

::::{tab-item} pip
:sync: pip

```{code-block} bash
pip show aeon  # see information about the installation i.e. version and file location
pip freeze  # see all installed packages for the current environment
```
::::

::::{tab-item} uv
:sync: uv

```{code-block} bash
uv pip show aeon  # see information about the installation i.e. version and file location
uv pip list  # see all installed packages for the current environment
```
::::

::::{tab-item} conda
:sync: conda

```{code-block} bash
conda list aeon  # see information about the installation i.e. version and file location
conda list  # see all installed packages for the current environment
```
::::

:::::

You are ready for the [getting started guide](getting_started.md).

## Optional Dependencies

Some estimators and functionality require optional dependencies. Without these
dependencies, you may find that you will be prompted to install an additional package
when trying to use certain functionality.

Installing all dependencies with `all_extras` (barring certain unstable ones) can take
a while to process and introduce limitations on the versioning of other packages, but
will allow all `aeon` functionality to be used without impediment.

For more information on the dependencies of `aeon` and more dependencies groups (such
as only dependencies for deep learning, or a list less stable dependencies excluded
from `all_extras`), see the
[`pyproject.toml`](https://github.com/aeon-toolkit/aeon/blob/main/pyproject.toml)
configuration file.

```{note}
The deep learning dependencies (`tensorflow` and `keras`) are not yet available for
Python 3.14, so `all_extras` does not install them for this version. Use Python
3.13 or lower if you need the deep learning estimators.
```

```{warning}
Some dependencies included in `all_extras` may have installation issues for macOS
with ARM processors. More details can be found in the troubleshooting section below.
```

(install-the-latest-development-version-using-pip)=
## Install the latest development version

The latest developments and bugfixes for `aeon` are available on the [`aeon`
GitHub](https://github.com/aeon-toolkit/aeon) `main` branch. This will include the
latest features and bug fixes, but can be more unstable than the latest release. Like
for a release, we recommend [creating an environment](#step-1-create-an-environment)
first.

:::::{tab-set}
:sync-group: tool

::::{tab-item} pip
:sync: pip

If you already have the latest `aeon` release or the `aeon` GitHub `main` branch
installed, you will have to uninstall it first:

```{code-block} bash
pip uninstall aeon
```

Then install the `main` branch, with core dependencies or with all stable
dependencies:

```{code-block} bash
pip install -U git+https://github.com/aeon-toolkit/aeon.git@main
pip install -U "aeon[all_extras] @ git+https://github.com/aeon-toolkit/aeon.git@main"
```
::::

::::{tab-item} uv
:sync: uv

Install the `main` branch, with core dependencies or with all stable dependencies.
This replaces a release of `aeon` if one is installed.

```{code-block} bash
uv pip install -U "aeon @ git+https://github.com/aeon-toolkit/aeon.git@main"
uv pip install -U "aeon[all_extras] @ git+https://github.com/aeon-toolkit/aeon.git@main"
```
::::

::::{tab-item} conda
:sync: conda

The development version is not available from conda-forge. Use the `pip` commands
inside your conda environment.
::::

:::::

The same information regarding the macOS ARM processor, checking the installation and
the `pyproject.toml` dependencies given above applies here as well.

## Troubleshooting

If the common errors below do not help, it may be worth checking out the [scikit-learn
troubleshooting section](https://scikit-learn.org/stable/install.html#troubleshooting)

(modulenotfounderror)=
:::{dropdown} `ModuleNotFoundError`
The most frequent reason for `ModuleNotFoundError` is installing `aeon` with
minimum dependencies (i.e. just `pip install aeon`) and using an estimator which
interfaces a package that has not been installed in the environment. To resolve this,
install the missing package, or install `aeon` with maximum dependencies (see above)
or install the individual packages as prompted by the error.
:::

(importerror)=
:::{dropdown} `ImportError`
Import errors are often caused by an improperly linked virtual environment. Make sure
that your environment is activated and linked to whatever IDE you are using. You can
find the instructions for doing so in VScode
[here](https://code.visualstudio.com/docs/python/environments). If you are using
Jupyter Notebooks, follow
[these instructions](https://janakiev.com/blog/jupyter-virtual-envs/) for adding your
virtual environment as a new kernel for your notebook.
:::

(installing-all-extras-on-mac-with-an-arm-processor)=
:::{dropdown} Installing `all_extras` on Mac with an ARM processor
If you are using a Mac with an ARM processor, you may encounter an error when installing
`aeon[all_extras]`. This is due to the fact that some libraries included in `all_extras`
are not compatible with ARM-based processors. If you encounter this issue, you can try
installing soft dependencies separately.

We would appreciate if you could report any issues you encounter with the `all_extras`
installation on ARM-based processors to the [aeon GitHub issues page](https://github.com/aeon-toolkit/aeon/issues).

Also, ARM-based processors can have issues when installing packages distributed as
source distributions instead of Python wheels. To avoid this issue when installing a
package, you can try installing it through `conda` or use a prior version of the package
that was distributed as a wheel.
:::

(no-matches-found-when-installing-all-extras)=
:::{dropdown} `no matches found` when installing `all_extras`
Some shells (i.e. the commonly used [Zsh](https://en.wikipedia.org/wiki/Z_shell)) use
square brackets as a special character. If you are using such a shell, you may
encounter this error when the dependency portion is not surrounded with quotes. The
commands of this page all use quotes:

```{code-block} bash
pip install -U "aeon[all_extras]"
```
:::
