# Developer Installation

Building the `aeon` package from source is a requirement for users who wish to
contribute to the `aeon` codebase and documentation. The following guide will walk
through downloading the latest development source code from GitHub and installing the
package.

Prior to these steps, we highly recommend creating a [virtual environment](../installation.md#using-a-pip-venv) for
the installation.

## Step 1 - Fork and/or clone the repository

The first step is to clone the `aeon` repository to a local directory. If you plan to
make a pull request on the GitHub repository, you should first fork the repository.
Create a fork of the `aeon` repository by clicking the "Fork" button in the top right
corner of the repository page or [here](https://github.com/aeon-toolkit/aeon/fork).

The `aeon` repository should be cloned to a local directory using [Git](https://git-scm.com/).

Using the `git` command line, the following commands will clone the `main` branch of
the repository to a local directory:

```{code-block} powershell
git clone https://github.com/aeon-toolkit/aeon.git
cd aeon
```

If you have forked the repository, clone your fork instead i.e.:

```{code-block} powershell
git clone https://github.com/your-username/aeon.git
cd aeon
```

## Step 2 - Building `aeon` from source

To build and install `aeon` from source, navigate to the local clone's root directory
and type:

```{code-block} powershell
pip install --editable .[dev]
```

Alternatively, the `.` may be replaced with a full or relative path to the root
directory.

This will install the `aeon` package in editable mode with dependencies required for
development. The `--editable` flag allows you to edit the code in-place and have the
changes reflected in the installed package without having to re-install the package.

If you need to work with optional dependencies, it you can also install the
`all_extras` extras:

```{code-block} powershell
pip install --editable .[dev,all_extras]
```

**If this results in a "no matches found" error**, it may be due to how your shell
handles special characters. Try surrounding the dependency portion with quotes:

```{code-block} powershell
pip install --editable ."[dev]"
```

## Step 3 - Install pre-commit

The `aeon` repository uses [pre-commit](https://pre-commit.com/) to run a series of checks on the codebase
before committing changes. To install pre-commit, run:

```{code-block} powershell
pre-commit install
```

This will run various code-quality hooks on the codebase before committing changes,
potentially changing the formatting of your code.

This is a requirement to make a pull request, and only in exceptional circumstances
will a pull request be accepted without passing pre-commit checks.
