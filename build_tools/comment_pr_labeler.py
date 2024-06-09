"""Labels PRs based on bot comment checkboxes."""

import json
import os
import sys

from github import Github

context_dict = json.loads(os.getenv("CONTEXT_GITHUB"))

repo = context_dict["repository"]
g = Github(os.getenv("GITHUB_TOKEN"))
repo = g.get_repo(repo)
issue_number = context_dict["event"]["issue"]["number"]
issue = repo.get_issue(number=issue_number)
comment_body = context_dict["event"]["comment"]["body"]
comment_user = context_dict["event"]["comment"]["user"]["login"]
labels = [label.name for label in issue.get_labels()]

if comment_user != "aeon-actions-bot[bot]":
    sys.exit(0)


def check_label_option(label, option):
    """Add or remove a label based on a checkbox in a comment."""
    if f"- [x] {option}" in comment_body:
        if label not in labels:
            issue.add_to_labels(label)
    elif f"- [ ] {option}" in comment_body:
        if label in labels:
            issue.remove_from_labels(label)


label_options = [
    ("full pre-commit", "Run `pre-commit` checks for all files"),
    ("full pytest actions", "Run all `pytest` tests and configurations"),
    ("full examples run", "Run all notebook example tests"),
    ("codecov actions", "Run numba-disabled `codecov` tests"),
    (
        "stop pre-commit fixes",
        "Stop automatic `pre-commit` fixes (always disabled for drafts)",
    ),
]

for option in label_options:
    check_label_option(option[0], option[1])
