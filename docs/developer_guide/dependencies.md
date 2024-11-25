# Dependencies

There are three types of dependencies in `aeon`: **core**, **soft**, or **developer**.

- **Core** dependencies are required to install and run `aeon` and are automatically
installed with `aeon` i.e.  `scikit-learn` and `numpy`
- **Soft** dependencies are only required to import certain modules, but not necessary
to use most functionalities. A soft dependency is not installed automatically with the
package unless an extra dependency set i.e. `all_extras` is used.
- **Developer** dependencies are required for `aeon` developers, but not for typical
users of `aeon` i.e. `pytest` and `pre-commit`. Documentation dependencies are also
included in this category.

We are unlikely to add new core dependencies, without a strong reason. Soft dependencies
should be the first choice for new dependencies, but ideally the code should be written
in `aeon` itself if possible.

Al dependencies are managed in the [`pyproject.toml`](https://github.com/aeon-toolkit/aeon/blob/main/pyproject.toml)
file following the [PEP 621](https://www.python.org/dev/peps/pep-0621/) convention.
Core dependencies are listed in the `dependencies` dependency set and
developer dependencies are listed in the `dev` and `docs` dependency sets.

## Adding a soft dependency

Soft dependencies in `aeon` should usually be restricted to the classes, functions
and/or modules which require them. Using any other part of the package should not
require the soft dependency.

Any new soft dependency needs to be added to the `all_extras` dependency set.
`unstable_extras` should be used instead if the dependency is unstable to install for
whatever reason i.e. it requires extra compilers to be installed or is only available
for specific operating systems. The vast majority of users should be able to install
`all_extras` without any issues.

Informative warnings or error messages for missing soft dependencies should be raised,
in a situation where a user would need them. This is handled through our
[`_check_soft_dependencies` utility](https://github.com/aeon-toolkit/aeon/blob/main/aeon/utils/validation/_dependencies.py).

There are specific conventions to add such warnings in estimators.
To add an estimator with a soft dependency, ensure the following:

- Imports of the soft dependency only happen inside the estimator, e.g., in `_fit` or
`__init__` methods of the estimator. In `__init__`, imports should happen only after
calls to `super(cls).__init__`.
- The `python_dependencies` tag of the estimator is populated with a `str`, or a `list`
of `str` for each dependency. Exceptions will automatically be raised when constructing
the estimator in an environment without the required packages.
- In a case where the package import differs from the package name, i.e.,
`import package_string` is different from `pip install different-package-string`
(usually the case for packages containing a dash in the name), the
`_check_soft_dependencies` utility should be used in `__init__`. Both the warning and
constructor call should use the `package_import_alias` argument for this.
- If the soft dependencies require specific python versions, the `python_version` tag
should also be populated, with a PEP 440 compliant version specification `str` such as
`"<3.10"` or `">3.6,~=3.8"`.
- Decorate all pytest tests that import soft dependencies with a
`@pytest.mark.skipif(...)` conditional on a check to `_check_soft_dependencies` for your
new soft dependency. This decorator will then skip your test unless the system has the
required packages installed.
