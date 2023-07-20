# Developer Installation

To install the latest development version of ``aeon``, or earlier versions, the sequence of steps is as follows:

Step 1 - ``git`` clone the ``aeon`` repository, the latest version or an earlier version.
Step 2 - ensure build requirements are satisfied
Step 3 - ``pip`` install the package from a ``git`` clone, with the ``editable`` parameter.

Detail instructions for all steps are given below.
For brevity, we discuss steps 1 and 3 first; step 2 is discussed at the end, as it will depend on the operating system.

## Step 1 - clone the git repository

The ``aeon`` repository should be cloned to a local directory, using a graphical user interface, or the command line.

Using the ``git`` command line, the sequence of commands to install the latest version is as follows:

```{code-block} powershell
    git clone https://github.com/aeon-toolkit/aeon.git
    cd aeon
    git checkout main
    git pull
```

To build a previous version, replace line 3 with:

```{code-block} powershell
    git checkout <VERSION>
```

This will checkout the code for the version ``<VERSION>``, where ``<VERSION>`` is a valid version string.
Valid version strings are the repository's ``git`` tags, which can be inspected by running ``git tag``.

You can also [download](https://github.com/aeon-toolkit/aeon/releases) a zip archive of the version from GitHub.

## Step 2 - satisfying build requirements

Before carrying out step 3, the ``aeon`` build requirements need to be satisfied.
Details for this differ by operating system, and can be found in the `aeon build requirements` section below.

Typically, the set-up steps needs to be carried out only once per system.

## Step 3 - building aeon from source

To build and install ``aeon`` from source, navigate to the local clone's root directory and type:

```{code-block} powershell
    pip install .
```

Alternatively, the ``.`` may be replaced with a full or relative path to the root directory.

For a developer install that updates the package each time the local source code is changed, install ``aeon`` in editable mode, via:

```{code-block} powershell
    pip install --editable .[dev]
```

This allows editing and extending the code in-place. See also
`pip reference on editable installs <https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`_).

```{note}
    You will have to re-run:

    pip install --editable .

    every time the source code of a compiled extension is changed (for
    instance when switching branches or pulling changes from upstream).
```

## Building binary packages and installers

The ``.whl`` package and ``.exe`` installers can be built with:

```{code-block} powershell
    pip install build
    python -m build --wheel
```

The resulting packages are generated in the ``dist/`` folder.


## aeon build requirements

This section outlines the ``aeon`` build requirements. These are required for:

* installing ``aeon`` from source, e.g., development versions
* the advanced developer set-up


## Setting up a development environment

First set up a new virtual environment. Our instructions will go through the commands to set up a ``conda`` environment which is recommended for aeon development.
This relies on an `anaconda installation <https://www.anaconda.com/products/individual#windows>`_. The process will be similar for ``venv`` or other virtual environment managers.

In the ``anaconda prompt`` terminal:

1. Navigate to your local aeon folder :code:`cd aeon`

2. Create new environment with python 3.8: :code:`conda create -n aeon-dev python=3.8`

   .. warning::
       If you already have an environment called "aeon-dev" from a previous attempt you will first need to remove this.

3. Activate the environment: :code:`conda activate aeon-dev`

4. Build an editable version of aeon :code:`pip install -e .[all_extras,dev]`

5. If everything has worked you should see message "successfully installed aeon"

Some users have experienced issues when installing NumPy, particularly version 1.19.4.

```{note}
    If step 4. results in a "no matches found" error, it may be due to how your shell handles special characters.

    - Possible solution: use quotation marks:

      pip install -e ."[all_extras,dev]"
```

```{note}
    Another option under Windows is to follow the instructions for `Unix-like OS`_, using the Windows Subsystem for Linux (WSL).
    For installing WSL, follow the instructions `here <https://docs.microsoft.com/en-us/windows/wsl/install-win10#step-2---check-requirements-for-running-wsl-2>`_.
```
