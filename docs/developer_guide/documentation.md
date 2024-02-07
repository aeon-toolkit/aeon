# Developing Documentation

`aeon`'s documentation standards include:

* Documenting code using NumPy docstrings and `aeon` conventions
* Following ``aeon``'s docstring convention for public code artifacts and modules
* Adding new public functionality to the [api_reference](https://www.aeon-toolkit.org/en/stable/api_reference.html) and [user_guide](https://www.aeon-toolkit.org/en/stable/getting_started.html).

More detailed information on ``aeon``'s documentation format is provided below.

## Docstring conventions

`aeon` uses the numpydoc Sphinx extension and follows [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).

To ensure docstrings meet expectations, `aeon` uses a combination of validations built into `numpydoc`, `pydocstyle pre-commit` checks (set to the NumPy convention) and automated testing of docstring examples to ensure the code runs without error. However, the automated docstring validation in pydocstyle only covers basic formatting Passing these tests is necessary to meet the `aeon` docstring conventions, but is not sufficient for doing so.

To ensure docstrings meet aeon's conventions, developers are expected to check their docstrings against numpydoc and `aeon` conventions and [reviewer's guide](https://www.aeon-toolkit.org/en/stable/contributing/reviewer_guide.html) are expected to also focus feedback on docstring quality.

## ``aeon`` specific conventions

Beyond basic NumPy docstring formatting conventions, developers should focus on:

- Ensuring all parameters (classes, functions, methods) and attributes (classes) are documented completely and consistently
- Including links to the relevant topics in the :ref:`glossary` or :ref:`user_guide` in the extended summary
- Including an `Examples` section that demonstrates at least basic functionality in all public code artifacts
- Adding a `See Also` section that references related `aeon` code artifacts as applicable
- Including citations to relevant sources in a `References` section

In many cases a parameter, attribute return object, or error may be described in many docstrings across aeon. To avoid confusion, developers should  make sure their docstrings are as similar as possible to existing docstring descriptions of the the same parameter, attribute, return object or error.

Accordingly, `aeon` estimators and most other public code artifcations should generally include the following NumPy docstring convention sections:

1. Summary
2. Extended Summary
3. Parameters
4. Attributes (classes only)
5. Returns or Yields (as applicable)
6. Raises (as applicable)
7. See Also (as applicable)
8. Notes (as applicable)
9. References (as applicable)
10. Examples

## Summary and extended summary

The summary should be a single line, followed by a (properly formatted) extended summary. The extended summary should include a user friendly explanation of the code artifacts functionality.

For all `aeon` estimators and other code artifacts that implement an algorithm, the extended summary should include a short, user-friendly synopsis of the algorithm being implemented. When the algorithm is implemented using multiple `aeon` estimators, the synopsis should first provide a high-level summary of the estimator components (e.g. transformer1 is applied then a classifier). Additional user-friendly details of the algorithm should follow (e.g. describe how the transformation and classifier work).

A developer can link to a particular area of the user guide by including an explicit cross-reference and following the steps for referencing in Sphinx (see the helpful description on [Sphinx cross-references](https://docs.readthedocs.io/en/stable/guides/cross-referencing-with-sphinx.html) posted by Read the Docs). Again developers are encouraged to add important content to the user guide and link to it if it does not already exist.

### See Also

This section should reference other `aeon` code artifcats related to the code artifact being documented by the docstring. Developers should use judgement in determining related code artifcats. For example, rather than listin all other performance metrics, a percentage error based performance metric
might only list other percentage error based performance metrics.  Likewise, a distance based classifier might list other distance based classifiers but
not include other types of time series classifiers.

### Notes

The notes section can include several types of information, including:

- Mathematical details of a code object or other important implementation details (using `..math` or `:math:` functionality)
- Links to alternative implementations of the code artifact that are external to `aeon` (e.g. the Java implementation of an `aeon` time series classifier)
- state changing methods (`aeon` estimator classes)

### References

`aeon` estimators that implement a concrete algorithm should generally include citations to the original research article, textbook or other resource
that describes the algorithm. Other code artifacts can include references as warranted (for example, references to relevant papers are included in
aeon's performance metrics).

This should be done by adding references into the references section of the docstring, and then typically linking to these in other parts of the docstring.

The references you intend to link to within the docstring should follow a very specific format to ensure they render correctly. See the example below. Note the space between the ".." and opening bracket, the space after the closing bracket, and how all the lines after the first line are aligned immediately with the opening bracket. Additional references should be added in exactly the same way, but the number enclosed in the bracket should be incremented.

```{code-block} powershell
.. [1] Some research article, link or other type of citation.
   Long references wrap onto multiple lines, but you need to
   indent them so they start aligned with opening bracket on first line.
```

To link to the reference labeled as `[1]`, you use `[1]_`. This only works within the same docstring. Sometimes this is not rendered correctly if the "[1]_" link is preceded or followed by certain characters. If you run into this issue, try putting a space before and following the `[1]_` link.

### Examples

Most code artifacts in `aeon` should include an examples section. At a minimum this should include a single example that illustrates basic functionality. The examples should use either a built-in `aeon` dataset or other simple data (e.g. randomly generated data, etc) generated using an `aeon` dependency (e.g. NumPy, pandas, etc) and whereever possible only depend on `aeon` or its core dependencies. Examples should also be designed to run quickly where possible. For quick running code artifacts, additional examples can be included to illustrate the affect of different parameter settings.

### Examples of Good `aeon` Docstrings

Here are a few examples of `aeon` code artifacts with good documentation.

#### Estimators

[BOSSEnsemble](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.classification.dictionary_based.BOSSEnsemble.html#aeon.classification.dictionary_based.BOSSEnsemble)

#### Functions
[dtw_distance](https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.distances.dtw_distance.html)

[numpydoc](https://numpydoc.readthedocs.io/en/latest/index.html)

[pydocstyle](http://www.pydocstyle.org/en/stable/)

[ContractableBOSS](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.classification.dictionary_based.ContractableBOSS.html#aeon.classification.dictionary_based.ContractableBOSS)

[MeanAbsoluteScaledError](https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.performance_metrics.forecasting.MeanAbsoluteScaledError.html)

[sphinx](https://www.sphinx-doc.org/)

[readthedocs](https://readthedocs.org/projects/aeon-toolkit/)

## Documentation Build

We use [sphinx](https://www.sphinx-doc.org/) to build our documentation and [readthedocs](https://readthedocs.org/projects/aeon-toolkit/) to host it. You can find our latest documentation [here](https://www.aeon-toolkit.org/en/latest/).

The source files can be found in [docs/](https://github.com/aeon-toolkit/aeon/tree/main/docs/). The main configuration file for sphinx is [conf.py](https://github.com/aeon-toolkit/aeon/blob/main/docs/conf.py) and the main page is [index.md](https://github.com/aeon-toolkit/aeon/blob/main/docs/index.md). To add new pages, you need to add a new `.rst` file and include it in the `index.md` file.

To build the documentation locally, you need to install a few extra dependencies listed in [pyproject.toml](https://github.com/aeon-toolkit/aeon/blob/main/pyproject.toml).
1. To install extra dependencies from the root directory, run:

```{code-block} powershell
pip install .[docs]
```

2. To build the website locally, run:

```{code-block} powershell
cd docs
make html
```

You may need to install pandoc to build the documentation locally.
