"""Writes a comment on PR opening.

Includes output from the labeler action.
"""

import json
import os

from github import Github

context_dict = json.loads(os.getenv("CONTEXT_GITHUB"))

repo = context_dict["repository"]
g = Github(context_dict["token"])
repo = g.get_repo(repo)
pr_number = context_dict["event"]["number"]
pr = repo.get_pull(number=pr_number)

pr.create_issue_comment("This is a test comment.")
