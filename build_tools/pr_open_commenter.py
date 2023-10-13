"""Writes a comment on PR opening.

Includes output from the labeler action.
"""

import json
import os
import sys

from github import Github

context_dict = json.loads(os.getenv("CONTEXT_GITHUB"))

repo = context_dict["repository"]
g = Github(context_dict["token"])
repo = g.get_repo(repo)
pr_number = context_dict["event"]["number"]
pr = repo.get_pull(number=pr_number)

print(sys.argv)  # noqa
title_labels = sys.argv[1]
title_labels_new = sys.argv[2]
content_labels = sys.argv[3]
content_labels_status = sys.argv[4]

title_labels_str = ""
if len(title_labels) == 0:
    title_labels_str = (
        "\nI did not find any labels to add based on the title. Please "
        "add the ENH, MNT, BUG, DOC and/or GOV tag to your pull "
        "requests titles. For now you can add the labels manually.\n"
    )
elif len(title_labels_new) != 0:
    title_labels_str = (
        "\nI have added the following labels to this PR based on the "
        f"title: {title_labels_new}."
    )
    if len(title_labels) != len(title_labels_new):
        title_labels_str += (
            " The following labels were already present: "
            f"{set(title_labels) - set(title_labels_new)}"
        )
    title_labels_str += "\n"


content_labels_str = ""
if len(content_labels) != 0:
    if content_labels_status == "used":
        content_labels_str = (
            "\nI have added the following labels to this PR based on "
            f"the changes: {content_labels}. Feel free to change "
            "these if they do not properly represent the PR.\n"
        )
    elif content_labels_status == "ignored":
        content_labels_str = (
            "\nI would have added the following labels to this PR "
            f"based on the changes: {content_labels}, however some "
            "package labels are already present.\n"
        )
    elif content_labels_status == "large":
        content_labels_str = (
            "\nThis PR changes to many different packages (3+) to "
            "automatically add labels, please manually add package "
            "labels if relevant.\n"
        )
elif title_labels_str == "":
    content_labels_str = (
        "\nI did not find any labels to add that did not already "
        "exist. If the content of your PR changes, make sure to "
        "update the labels accordingly.\n"
    )

pr.create_issue_comment(
    f"""
    Thank you for contributing to `aeon`!
    {title_labels_str}
    {content_labels_str}
    The 'Checks' tab above or the panel below will show the status of our automated
    tests. From the panel below you can click on 'Details' to the right of the check
    to see more information if it fails. If our `pre-commit` code quality check fails,
    any trivial fixes will automatically be pushed to your PR.

    Don't hesitate to ask questions on the `aeon` Slack channel if you have any!
    """
)
