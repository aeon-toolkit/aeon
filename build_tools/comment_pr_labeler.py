"""Labels PRs based on bot comment checkboxes."""

import json
import os
import re
import sys

from github import Github

context_dict = json.loads(os.getenv("CONTEXT_GITHUB"))

repo = context_dict["repository"]
g = Github(os.getenv("GITHUB_TOKEN"))
repo = g.get_repo(repo)
issue_number = context_dict["event"]["issue"]["number"]
issue = repo.get_issue(number=issue_number)
comment_body = context_dict["event"]["comment"]["body"]

if issue:
    pr.add_to_labels("documentation", "no changelog")
    sys.exit(0)
elif "[bot]" in pr.user.login:
    sys.exit(0)

if comment_body.startswith("@aeon-actions-bot") and "assign" in comment_body.lower():
    mentioned_users = re.findall(r"@[a-zA-Z0-9_-]+", comment_body)
    mentioned_users = [user[1:] for user in mentioned_users[1:]]

    for user in mentioned_users:
        issue.add_to_assignees(user)
