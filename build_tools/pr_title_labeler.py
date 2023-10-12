"""Labels PRs based on title.

Must be run in a github action with the pull_request_target event.

Based on the scikit-learn v1.3.1 label_title_regex.py script.
"""

import json
import os
import re

from github import Github

context_dict = json.loads(os.getenv("CONTEXT_GITHUB"))

repo = context_dict["repository"]
g = Github(context_dict["token"])
repo = g.get_repo(repo)
pr_number = context_dict["event"]["number"]
issue = repo.get_issue(number=pr_number)
title = issue.title

regex_to_labels = [
    (r"\bENH\b", "enhancement"),
    (r"\bMNT\b", "maintenance"),
    (r"\bBUG\b", "bug"),
    (r"\bDOC\b", "documentation"),
    (r"\bGOV\b", "governance"),
]

labels_to_add = [label for regex, label in regex_to_labels if re.search(regex, title)]

if labels_to_add:
    issue.add_to_labels(*labels_to_add)

with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
    print(f"new-labels={labels_to_add}", file=fh)  # noqa: T201
