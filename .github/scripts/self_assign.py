"""
Script for the Self-Assign Bot.

It checks if a comment on an issue or PR includes the trigger
phrase(as defined) and a mentioned user.
If it does, it assigns the issue/PR to the mentioned user.
"""

import os
import re

from github import Github

# Initialize a Github instance:
g = Github(os.getenv("GITHUB_TOKEN"))

# Get the repo from environment variables
repo = g.get_repo(os.getenv("GITHUB_REPOSITORY"))

# Get the issue from the payload
issue_number = int(os.getenv("ISSUE_NUMBER"))
issue = repo.get_issue(number=issue_number)

# Get the comment from the payload
comment_body = os.getenv("COMMENT_BODY")

# Check if the comment contains the word "assign"
if "assign" in comment_body.lower():
    # Extract the username mentioned in the comment
    mentioned_users = re.findall(r"@([a-zA-Z0-9](?:-?[a-zA-Z0-9]){0,38})", comment_body)
    for mentioned_user in mentioned_users:
        # Assign the Issue/PR to the user
        issue.add_to_assignees(mentioned_user)
