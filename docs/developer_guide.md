# Developer Guide

Welcome to the `aeon` developer guide. This guide is intended for new developers who
want to start contributing code to `aeon` and current developers who want to learn
about specific topics for code and non-code developments.

The following is a quick checklist for new developers. At any point during the process,
feel free to post questions on Slack or ask [core developers](./about.md#core-developers)
for help.

- Fork the repository and install a [development version](developer_guide/dev_installation.md)
of `aeon`.
- Get familiar with the [git workflow](developer_guide/git_workflow.rst).
- Set up [Continuous Integration (CI)](developer_guide/continuous_integration.rst)
testing locally and learn how to check the status on pull requests.
- Read up on the `aeon` [coding standards](developer_guide/coding_standards.rst) and
pre-commit setup.

`aeon` follows the `scikit-learn` API whenever possible. If you’re new to
`scikit-learn`, take a look at their [getting-started guide](https://scikit-learn.org/stable/getting_started.html).
If you’re already familiar with `scikit-learn`, you may still learn something new from
their [developer's guide](https://scikit-learn.org/stable/developers/index.html).

## Specific Topics

Below we list further reading and guidance for specific topics in the development of
`aeon`.

```{toctree}
:maxdepth: 1

developer_guide/add_dataset
developer_guide/add_estimators
developer_guide/coding_standards
developer_guide/continuous_integration
developer_guide/dependencies
developer_guide/deprecation.md
developer_guide/dev_installation.md
developer_guide/documentation
developer_guide/git_workflow
developer_guide/release.md
developer_guide/testing_framework
```
