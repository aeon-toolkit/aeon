"""Script for the GitHub issue self-assign bot.

It checks if a comment on an issue or PR includes the trigger
phrase (as defined) and a mentioned user.
If it does, it assigns the issue/PR to the mentioned user.
"""

import json
import os
import re

from github import Github

context_dict = json.loads(os.getenv("CONTEXT_GITHUB"))

repo = context_dict["repository"]
g = Github(os.getenv("GITHUB_TOKEN"))
repo = g.get_repo(repo)
issue_number = context_dict["event"]["issue"]["number"]
issue = repo.get_issue(number=issue_number)
comment_body = context_dict["event"]["comment"]["body"]

# Assign tagged used to the issue if the comment includes the trigger phrase
# and starts with a mention to the bot
if comment_body.startswith("@aeon-actions-bot") and "assign" in comment_body.lower():
    mentioned_users = re.findall(r"@[a-zA-Z0-9_-]+", comment_body)
    mentioned_users = [user[1:] for user in mentioned_users[1:]]

    for user in mentioned_users:
        issue.add_to_assignees(user)
