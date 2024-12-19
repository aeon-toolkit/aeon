# Contributing to aeon

`aeon` is a community-driven project and contributions are most welcome. We value all
kinds of contributions, not just code. Improvements to docs, bug reports, and taking
on communications or code of conduct responsibilities are all examples of valuable
contributions beyond code which help make `aeon` a great package.

Please consider whether you will be able to tackle and issue or pull request before
assigning yourself to it. If the issue requires editing Python code, you should have
some experience with Python and be able to run tests. If the issue tackles the
specifics of a machine learning algorithm, some relevant knowledge of machine learning
will be required. While we want to encourage new contributors, a base level
of knowledge is required to make a meaningful contribution to certain issues.
ChatGPT is not a replacement for this knowledge.

Pull requests from unknown contributors which do not attempt to resolve the issue being
addressed, completely disregard the pull request template, or consist of low quality AI
generated output may be closed without review.

In the following we will give a brief overview of how to contribute to `aeon`. Making
contributions to open-source projects takes a bit of proactivity and can be daunting at
first, but members of the community are here to help and answer questions. If you get
stuck, please don’t hesitate to talk with us or raise an issue.

Recommended steps for first time contributors, or to get started with regular
contributions:

1. Say hello in the `introductions` or `contributors` channel on [Slack](https://join.slack.com/t/aeon-toolkit/shared_invite/zt-22vwvut29-HDpCu~7VBUozyfL_8j3dLA)
and mention which places you are interested in contributing to.
2. Get setup for development, see the [developer install instructions](developer_guide/dev_installation.md)
for creating a fork of `aeon`.
3. Pick an `enhancement`, `documentation` or `maintenance` issue from the [issue list](https://github.com/aeon-toolkit/aeon/issues)
to complete i.e. improving an algorithm, docstring or test. The [good first issue](https://github.com/aeon-toolkit/aeon/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
list may be a good place to start.
4. Post on the issue which you want to work on, so that others know you are working on
it. **First ensure that the issue is not already being worked on. Look if there are any
linked PRs and search the issue number in the pull requests list.**
To assign yourself an **Issue/Pull Request**, please post a comment in the issue
including '@aeon-actions-bot', the username of people to assign and the word `assign`:

    For example:
    ```python
    @aeon-actions-bot assign @MatthewMiddlehurst
    ```
    Please ensure you understand and have a plan for resolving the issue before
assigning yourself. Feel free to ask for clarification if you are unsure. If it is a
larger issue with multiple components, indicate which part you are working on. A Core
Developer may suggest a different issue if the one you chose is complex or somebody is
already working on it.
5. Create a [pull request (PR)](https://github.com/aeon-toolkit/aeon/compare)
with your changes from your fork. For help, see the [GitHub documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)
or ask in Slack. Follow the PR template, template comments and checklist. Please make
sure to include an appropriate [title tag](contributing/issues.md). **Do not just delete the PR template
text.**
6. A Core Developer will review your PR and may provide feedback, which you can then
address. If you are unsure about any feedback, please ask for clarification. Please
be patient, as Core Developers are volunteers and may be busy with other tasks or life
outside the package. It could take a while to get a review during
slow periods, so please do not rush to @ developers or repeatedly ask for a review.
Consider opening the PR as a draft until it is ready for review and passing tests.
8. Once your PR is approved, it will be merged into the `aeon` repository. Thanks for
making a contribution! Make sure you are included in the [list of contributors](contributors.md).

Further guidance for contributing to `aeon` via GitHub can be found on the
[developer guide](developer_guide.md). It is not necessary to read everything here prior to
contributing, but if your issue to related to a specific topic i.e. documentation or
testing you may find it useful.

If your intended method of contribution does not fit into the above steps, please
reach out to us on [Slack](https://join.slack.com/t/aeon-toolkit/shared_invite/zt-22vwvut29-HDpCu~7VBUozyfL_8j3dLA)
for discussion. While GitHub contributions are the most common, it is not the only
way to contribute to `aeon`.

## Acknowledging Contributions

We follow the [all-contributors specification](https://allcontributors.org) and
recognise various types of contributions. Take a look at our past and current
[contributors](contributors.md)!

If you are a new contributor, make sure we add you to our list of contributors. All
contributions are recorded in [.all-contributorsrc](https://github.com/aeon-toolkit/aeon/blob/main/.all-contributorsrc).
Alternatively, you can use the [@all-contributors](https://allcontributors.org/docs/en/bot/usage)
bot to do this for you. If the contribution is contained in a PR, please only @ the bot
when the PR has been merged. A list of relevant tags can be found [here](https://allcontributors.org/docs/en/emoji-key).

## Further Reading

For further information on contributing to `aeon`, please see the following pages.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card}
:text-align: center

Developer Guide

^^^

Guidance for `aeon` developers on a range of topics.

+++

```{button-ref} developer_guide
:color: primary
:click-parent:
:expand:

Developer Guide
```

:::

:::{grid-item-card}
:text-align: center

Opening Issues

^^^

Guidance for issues and reporting bugs in `aeon`.

+++

```{button-ref} contributing/reporting_bugs
:color: primary
:click-parent:
:expand:

Opening Issues
```

:::

:::{grid-item-card}
:text-align: center

Mentoring and Projects

^^^

`aeon` projects and mentoring opportunities.

+++

```{button-ref} mentoring
:color: primary
:click-parent:
:expand:

Mentoring and Projects
```
:::

::::

```{toctree}
:hidden:

contributing/issues.md
```
