"""Removes the good first issue tag when an issue has been assigned."""

import json
import os

from github import Github

context_dict = json.loads(os.getenv("CONTEXT_GITHUB"))

repo = context_dict["repository"]
g = Github(os.getenv("GITHUB_TOKEN"))
repo = g.get_repo(repo)
issue_number = context_dict["event"]["issue"]["number"]
issue = repo.get_issue(number=issue_number)
labels = [label.name for label in issue.get_labels()]

if "good first issue" in labels:
    issue.remove_from_labels("good first issue")
