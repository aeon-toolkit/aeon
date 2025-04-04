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
commenter = context_dict["event"]["comment"]["user"]["login"]

ASSIGNMENT_PHRASES = [
    "assign me",
    "please assign me",
    "can you assign this to me",
    "i want to work on this",
    "give me this issue",
]


body = comment_body.lower()
if "@aeon-actions-bot" in body:
    # Assign commenter if comment includes any phrase that is in ASSIGNMENT_PHRASES
    if any(phrase in body for phrase in ASSIGNMENT_PHRASES):
        issue.add_to_assignees(commenter)
    # Assign tagged used to the issue if the comment includes the trigger phrase
    elif "assign" in body:
        mentioned_users = re.findall(r"@[a-zA-Z0-9_-]+", comment_body)
        mentioned_users = [user[1:] for user in mentioned_users]
        mentioned_users.remove("aeon-actions-bot")

        for user in mentioned_users:
            issue.add_to_assignees(user)
