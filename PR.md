<!--
Thanks for contributing a pull request! Please ensure you have taken a look at our
contribution guide: https://www.aeon-toolkit.org/en/latest/contributing.html.

Feel free to delete sections of this template if they do not apply to your PR,
avoid submitting a blank template or empty sections.
If you are a new contributor, do not delete this template without a suitable
replacement or reason. If in doubt, ask for help. We're here to help!

Please be aware that we are a team of volunteers so patience is
necessary when waiting for a review or reply. There may not be a quick turnaround for
reviews during slow periods. While we value all contributions big or small, pull
requests which do not follow our guidelines may be closed.
-->

#### Reference Issues/PRs

<!--
Example: Fixes #1234. See also #3456.
Please use keywords (e.g., Fixes) to create link to the issues or pull requests
you resolved, so that they will automatically be closed when your pull request
is merged. See https://github.com/blog/1506-closing-issues-via-pull-requests
-->

- None.

#### What does this implement/fix? Explain your changes.

<!--
A clear and concise description of what you have implemented.
-->

- Refactors pointwise distance functions to use shared factory helpers for 1D/2D handling and pairwise wrappers, reducing repeated boilerplate while keeping public APIs stable.
- Simplifies Minkowski’s pairwise core usage to align with the factory pattern.

#### Does your contribution introduce a new dependency? If yes, which one?

<!--
If your contribution does add a dependency, we may suggest adding it as an
optional/soft dependency to keep external dependencies of the core aeon package
to a minimum.
-->

- No.

#### Any other comments?

<!--
Any other information that is important to this PR or helpful for reviewers.
-->

**Performance comparison (pairwise only, pointwise distances)**
- Benchmarks run on main vs this branch for `squared`, `euclidean`, `manhattan`, `minkowski`
- Lengths: 100, 200
- Instances: 100, 250, 500
- Channels: 1, 5
- Iterations: 5

**Figures (placeholders)**
- Figure 1: `artifacts/pointwise-bench/pointwise_pairwise_channels_1.png` (main vs PR, channels=1)
- Figure 2: `artifacts/pointwise-bench/pointwise_pairwise_channels_5.png` (main vs PR, channels=5)

**Condensed pairwise performance summary (main vs PR)**
| distance | mean_change_pct | median_change_pct | min_change_pct | max_change_pct | faster | similar | slower |
|---|---:|---:|---:|---:|---:|---:|---:|
| euclidean | -0.42% | +0.30% | -18.34% | +10.15% | 2 | 8 | 2 |
| manhattan | +3.80% | +2.93% | -4.62% | +11.76% | 0 | 7 | 5 |
| minkowski | -13.67% | -12.53% | -19.71% | -10.35% | 12 | 0 | 0 |
| squared | -0.17% | +0.05% | -7.00% | +6.03% | 2 | 9 | 1 |

Full per-case table available in `artifacts/pointwise-bench/pointwise_pairwise_comparison.md`.

**Code reduction**
- Pointwise distance files total: 791 → 551 lines (−240 lines, −30.34%)
- With the new factory file included: 791 → 700 lines (net −91 lines, −11.50%)

### PR checklist

<!--
Please go through the checklist below. Please feel free to remove points if they are
not applicable. To check a box, replace the space inside the square brackets with an
'x' i.e. [x].
-->

##### For all contributions
- [ ] I've added myself to the [list of contributors](https://github.com/aeon-toolkit/aeon/blob/main/.all-contributorsrc). Alternatively, you can use the [@all-contributors](https://allcontributors.org/docs/en/bot/usage) bot to do this for you **after** the PR has been merged.
- [ ] The PR title starts with either [ENH], [MNT], [DOC], [BUG], [REF], [DEP] or [GOV] indicating whether the PR topic is related to enhancement, maintenance, documentation, bugs, refactoring, deprecation or governance.

##### For new estimators and functions
- [ ] I've added the estimator/function to the online [API documentation](https://www.aeon-toolkit.org/en/latest/api_reference.html).
- [ ] (OPTIONAL) I've added myself as a `__maintainer__` at the top of relevant files and want to be contacted regarding its maintenance. Unmaintained files may be removed. This is for the full file, and you should not add yourself if you are just making minor changes or do not want to help maintain its contents.

##### For developers with write access
- [ ] (OPTIONAL) I've updated aeon's [CODEOWNERS](https://github.com/aeon-toolkit/aeon/blob/main/CODEOWNERS) to receive notifications about future changes to these files.


<!--
Thanks for contributing!
-->

This pull request includes code written with the assistance of AI.
The code has **not yet been reviewed** by a human.
