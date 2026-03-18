"""Uncheck relevant boxes after running the comment edit workflow."""

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

if (
    "[bot]" in context_dict["event"]["sender"]["login"]
    or issue.pull_request is None
    or comment_user != "aeon-actions-bot[bot]"
    or "## Thank you for contributing to `aeon`" not in comment_body
):
    sys.exit(0)

pr = issue.as_pull_request()
comment = pr.get_issue_comment(context_dict["event"]["comment"]["id"])

new_comment_body = comment_body

if "- [x] Regenerate expected results for testing" in comment_body:
    new_comment_body = new_comment_body.replace(
        "- [x] Regenerate expected results for testing",
        "- [ ] Regenerate expected results for testing",
    )

if "- [x] Push an empty commit to re-run CI checks" in comment_body:
    new_comment_body = new_comment_body.replace(
        "- [x] Push an empty commit to re-run CI checks",
        "- [ ] Push an empty commit to re-run CI checks",
    )

if new_comment_body != comment_body:
    comment.edit(new_comment_body)
