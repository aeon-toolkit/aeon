"""Labels PRs and process based on bot comment checkboxes."""

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

if (
    issue.pull_request is None
    or comment_user != "aeon-actions-bot[bot]"
    or "## Thank you for contributing to `aeon`" not in comment_body
):
    with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
        print("empty_commit=false", file=fh)  # noqa: T201
    sys.exit(0)
pr = issue.as_pull_request()


def check_label_option(label, option):
    """Add or remove a label based on a checkbox in a comment."""
    if f"- [x] {option}" in comment_body:
        if label not in labels:
            pr.add_to_labels(label)
    elif f"- [ ] {option}" in comment_body:
        if label in labels:
            pr.remove_from_labels(label)


label_options = [
    ("full pre-commit", "Run `pre-commit` checks for all files"),
    ("run typecheck test", "Run `mypy` typecheck tests"),
    ("full pytest actions", "Run all `pytest` tests and configurations"),
    ("full examples run", "Run all notebook example tests"),
    ("codecov actions", "Run numba-disabled `codecov` tests"),
    (
        "stop pre-commit fixes",
        "Stop automatic `pre-commit` fixes (always disabled for drafts)",
    ),
    ("no numba cache", "Disable numba cache loading"),
]

for option in label_options:
    check_label_option(option[0], option[1])

repo_name = pr.head.repo.full_name
branch_name = pr.head.ref
with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
    print(f"repo={repo_name}", file=fh)  # noqa: T201
    print(f"branch={branch_name}", file=fh)  # noqa: T201

if "- [x] Push an empty commit to re-run CI checks" in comment_body:
    for comment in pr.get_comments():
        if (
            comment.user.login == comment_user
            and "## Thank you for contributing to `aeon`" in comment.body
        ):
            comment.edit(
                comment_body.replace(
                    "- [x] Push an empty commit to re-run CI checks",
                    "- [ ] Push an empty commit to re-run CI checks",
                )
            )
            break
    with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
        print("empty_commit=true", file=fh)  # noqa: T201
else:
    with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
        print("empty_commit=false", file=fh)  # noqa: T201
