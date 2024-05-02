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

# Define trigger phrases
trigger_phrases = [
    "aeon-assign bot",
    "assign this to",
    "please assign",
    "assign",
    "assigning",
]
trigger_pattern = "|".join(trigger_phrases)

# Check if the comment includes the trigger phrase
if comment_body.startswith("@aeon-actions-bot") and re.search(trigger_pattern, comment_body, re.IGNORECASE):
    # Extract the username mentioned in the comment
    mentioned_user_match = re.search(
        r"@([a-zA-Z0-9](?:-?[a-zA-Z0-9]){0,38})", comment_body
    )
    if mentioned_user_match:
        mentioned_user = mentioned_user_match.group(1)
        # Assign the Issue/PR to the user
        issue.add_to_assignees(mentioned_user)
