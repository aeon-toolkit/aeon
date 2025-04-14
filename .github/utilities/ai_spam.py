"""Script for handling AI Spam label on pull requests.

Triggered when AI Spam label is added to a PR,
it adds a comment and closes the PR.
"""

import json
import os
from github import Github

context_dict = json.loads(os.getenv("CONTEXT_GITHUB"))

repo_name = context_dict["repository"]
g = Github(os.getenv("GITHUB_TOKEN"))
repo = g.get_repo(repo_name)
pr_number = context_dict["event"]["pull_request"]["number"]
pr = repo.get_pull(pr_number)
label_name = context_dict["event"]["label"]["name"]

if label_name == 'AI Spam':
    comment_body = (
    f"This pull request has been flagged with the **AI Spam** label.\n\n"
    f"This PR is being closed."
    )
    pr.create_issue_comment(comment_body)
    pr.edit(state="closed")


