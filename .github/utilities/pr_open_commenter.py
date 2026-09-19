"""Writes a comment on PR opening.

Includes output from the labeler action.
"""

import json
import os
import sys
from urllib.parse import quote

from _commons import label_options
from github import Github

context_dict = json.loads(os.getenv("CONTEXT_GITHUB"))

repo = context_dict["repository"]
g = Github(os.getenv("GITHUB_TOKEN"))
repo = g.get_repo(repo)
pr_number = context_dict["event"]["number"]
pr = repo.get_pull(number=pr_number)

if "[bot]" in pr.user.login:
    sys.exit(0)

title_labels = os.getenv("TITLE_LABELS")[1:-1].replace("'", "").split(",")
title_labels_new = os.getenv("TITLE_LABELS_NEW")[1:-1].replace("'", "").split(",")
content_labels = os.getenv("CONTENT_LABELS")[1:-1].replace("'", "").split(",")
content_labels_status = os.getenv("CONTENT_LABELS_STATUS")

replacement_labels = [
    ("anomalydetection", "anomaly detection"),
    ("similaritysearch", "similarity search"),
]
for i, label in enumerate(content_labels):
    for cur_label, new_label in replacement_labels:
        if label == cur_label:
            content_labels[i] = new_label


def label_badge(label, color):
    """Create a badge displaying a label name in the label colour.

    The label colour is used as the badge background, with shields.io picking a
    readable text colour for it. Using it as the text colour instead makes some
    labels unreadable on one of the GitHub themes.
    """
    # shields.io reserves these characters in the badge name
    name = quote(label.replace("-", "--").replace("_", "__"))
    return f"![{label}](https://img.shields.io/badge/{name}-{color})"


labels = [(label.name, label.color) for label in repo.get_labels()]
title_labels = [label_badge(n, c) for n, c in labels if n in title_labels]
title_labels_new = [label_badge(n, c) for n, c in labels if n in title_labels_new]
content_labels = [label_badge(n, c) for n, c in labels if n in content_labels]

title_labels_str = ""
if len(title_labels) == 0:
    title_labels_str = (
        "I did not find any labels to add based on the title. Please "
        "add the [ENH], [MNT], [BUG], [DOC], [REF], [DEP] and/or [GOV] tags to your "
        "pull requests titles. For now you can add the labels manually."
    )
elif len(title_labels_new) != 0:
    arr_str = " ".join(title_labels_new)
    title_labels_str = (
        f"I have added the following labels to this PR based on the title: {arr_str}."
    )
    if len(title_labels) != len(title_labels_new):
        arr_str = " ".join(sorted(set(title_labels) - set(title_labels_new)))
        title_labels_str += f" The following labels were already present: {arr_str}."

content_labels_str = ""
if len(content_labels) != 0:
    if content_labels_status == "used":
        arr_str = " ".join(content_labels)
        content_labels_str = (
            "I have added the following labels to this PR based on "
            f"the changes made: {arr_str}. Feel free "
            "to change these if they do not properly represent the PR."
        )
    elif content_labels_status == "ignored":
        arr_str = " ".join(content_labels)
        content_labels_str = (
            "I would have added the following labels to this PR "
            f"based on the changes made: {arr_str}, "
            "however some package labels are already present."
        )
    elif content_labels_status == "large":
        content_labels_str = (
            "This PR changes too many different packages (>3) for "
            "automatic addition of labels, please manually add package "
            "labels if relevant."
        )
elif title_labels_str == "":
    content_labels_str = (
        "I did not find any labels to add that did not already "
        "exist. If the content of your PR changes, make sure to "
        "update the labels accordingly."
    )

comment_body = f"""
## Thank you for contributing to `aeon`

{title_labels_str}
{content_labels_str}

The [Checks](https://github.com/aeon-toolkit/aeon/pull/{pr_number}/checks) tab will show the status of our automated tests. You can click on individual test runs in the tab or "Details" in the panel below to see more information if there is a failure.

If our `pre-commit` code quality check fails, please run `pre-commit` locally and push the fixes to your PR branch.

Don't hesitate to ask questions on the `aeon` [Discord](https://discord.gg/D6rzqHGKRJ) channel if you have any.

<details><summary>PR CI actions</summary>
<p>

These checkboxes will add labels to enable or disable CI functionality for this PR. This may not take effect immediately, and a new commit may be required to run the new configuration.

- [ ] Run `pre-commit` checks for all files
- [ ] Run `mypy` typecheck tests
- [ ] Run all `pytest` tests and configurations
- [ ] Run all notebook example tests
- [ ] Run numba-disabled `codecov` tests
- [ ] Disable numba cache loading
- [ ] Regenerate expected results for testing
- [ ] Push an empty commit to re-run CI checks

</p>
</details>
    """  # noqa


def tick_label_boxes(body):
    """Tick the CI action boxes for labels currently on the PR."""
    pr_labels = [label.name for label in pr.get_labels()]
    for label, option in label_options:
        if label in pr_labels:
            body = body.replace(f"- [ ] {option}", f"- [x] {option}")
    return body


# the label edit workflow cannot tick boxes for labels added before this comment
# exists, as it has no comment to find. labels added while this workflow runs are
# caught by the second check after the comment is created
comment = pr.create_issue_comment(tick_label_boxes(comment_body))

new_comment_body = tick_label_boxes(comment_body)
if new_comment_body != comment.body:
    comment.edit(new_comment_body)
