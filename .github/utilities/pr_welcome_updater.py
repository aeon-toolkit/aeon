"""Labels PRs based on bot comment checkboxes."""

import json
import os
import sys

from _commons import label_options
from github import Github

context_dict = json.loads(os.getenv("CONTEXT_GITHUB"))

repo = context_dict["repository"]
g = Github(os.getenv("GITHUB_TOKEN"))
repo = g.get_repo(repo)
pr_number = context_dict["event"]["number"]
pr = repo.get_pull(number=pr_number)
labels = [label.name for label in pr.get_labels()]

if "[bot]" in context_dict["event"]["sender"]["login"]:
    sys.exit(0)

comment = None
for c in pr.get_issue_comments():
    if (
        c.user.login == "aeon-actions-bot[bot]"
        and "## Thank you for contributing to `aeon-eval`" in c.body
    ):
        comment = c
        break

if comment is None:
    sys.exit(0)

comment_body = comment.body
for option in label_options:
    if f"- [x] {option[1]}" in comment_body and option[0] not in labels:
        comment_body = comment_body.replace(
            f"- [x] {option[1]}",
            f"- [ ] {option[1]}",
        )
    elif f"- [ ] {option[1]}" in comment_body and option[0] in labels:
        comment_body = comment_body.replace(
            f"- [ ] {option[1]}",
            f"- [x] {option[1]}",
        )

comment.edit(comment_body)
