# Referencing Externally Written Code

When contributing code to `aeon` that is adapted, inspired by, or directly copied
from external sources, it is important to properly acknowledge the original authors
and include any necessary license information. This applies to code from other
repositories, research papers, Stack Overflow answers, or any other external source.

Proper attribution is both a legal requirement (to comply with open-source licenses)
and a matter of good practice in the open-source community.

## When is attribution required?

Attribution is required whenever code in `aeon`:

- Is directly copied or closely adapted from another repository or codebase
- Is inspired by or based on an existing implementation, even if significantly rewritten
- Implements an algorithm from a research paper using code provided by the authors
- Is adapted from a Stack Overflow answer or similar community resource

## How to add attribution

Attribution should be added in the **docstring** of the class or function containing
the external code. The correct place is in the `Notes` section of the docstring.

The format should include:

1. A description of how the code was used (e.g. "Adapted from", "Directly copied from",
"Inspired by")
2. A link to the original source
3. Copyright information and license (if available)

### Examples

**Code adapted from another repository:**

```python
Notes
-----
Adapted from the statsmodels 0.14.4 implementation
https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/filters/bk_filter.py
Copyright (c) 2009-2018 statsmodels Developers, BSD-3
```

**Code adapted from a repository by the original author:**

```python
Notes
-----
Adapted from the implementation from Ismail-Fawaz et. al
https://github.com/MSD-IRIMAS/LITE
by the code owner.
```

**Code adapted from multiple sources:**

```python
Notes
-----
This code was adapted from the tslearn and pyts functions.
pyts code:
https://pyts.readthedocs.io/en/latest/_modules/pyts/metrics/dtw.html
Copyright (c) 2018, Johann Faouzi and pyts contributors, BSD-3
tslearn code (line 974):
https://github.com/tslearn-team/tslearn/blob/main/tslearn/metrics/dtw_variants.py
Copyright (c) 2017, Romain Tavenard, BSD-2
```

**Code used with explicit permission from the owner:**

```python
Notes
-----
Parts of the code are adapted from https://github.com/hfawaz/cd-diagram
with permission from the owner.
```

**Code adapted from a class-level implementation (in the class docstring):**

```python
class MyEstimator:
    """My estimator.

    ...

    Notes
    -----
    Parts (i.e. get_params and set_params) adapted from the scikit-learn 1.5.0
    ``_BaseComposition`` class in utils/metaestimators.py.
    https://github.com/scikit-learn/scikit-learn/
    Copyright (c) 2007-2024 The scikit-learn developers, BSD-3
    """
```

## File-level attribution

If an entire file is adapted from an external source, attribution should be added
at the top of the file in the module docstring. For example:

```python
"""Utility methods to print system info for debugging.

Notes
-----
Adapted from the scikit-learn 1.5.0 show_versions function.
https://github.com/scikit-learn/scikit-learn/
Copyright (c) 2007-2024 The scikit-learn developers, BSD-3
"""
```

## License compatibility

`aeon` uses the [BSD-3-Clause license](https://github.com/aeon-toolkit/aeon/blob/main/LICENSE).
When incorporating external code, you must ensure the source license is compatible
with BSD-3-Clause. The following licenses are generally compatible:

- BSD-2-Clause
- BSD-3-Clause
- MIT License

If you are unsure whether a license is compatible, please raise the question in your
pull request or ask on the `aeon` [Discord](https://discord.com/invite/54ACzaFsnSA)
before submitting.

Code under licenses such as GPL is **not compatible** with `aeon`'s BSD-3-Clause
license and should not be incorporated.

## Summary checklist

Before submitting a pull request that includes externally written code, ensure:

- [ ] Attribution is added in the `Notes` section of the relevant docstring
- [ ] The original source URL is included
- [ ] Copyright information and license are included where available
- [ ] The source license is compatible with BSD-3-Clause
- [ ] If adapted from a paper's own implementation, this is noted explicitly
