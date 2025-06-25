"""Script for the GitHub issue self-assign bot.

It checks if a comment on an issue or PR includes the trigger
phrase (as defined) and a mentioned user.
If it does, it assigns the issue to the mentioned user.

Users without write access can only have up to 2 open issues assigned.
Users with write access (or admin) are exempt from this limit.
If a non-write user already has 2 or more open issues, the bot
comments on the issue with links to the currently assigned open issues.
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
issue_labels = {label.name.lower() for label in issue.labels}
pr = context_dict["event"]["issue"].get("pull_request")
comment_body = context_dict["event"]["comment"]["body"]
commenter = context_dict["event"]["comment"]["user"]["login"]

restricted_labels = {"meta-issue"}

# Assign tagged used to the issue if the comment includes the trigger phrase
body = comment_body.lower()
if "@aeon-actions-bot" in body and "assign" in body and not pr:
    # Check if the issue has any restricted labels for auto assignment
    label_intersect = issue_labels & restricted_labels
    if len(label_intersect) > 0:
        issue.create_comment(
            f"This issue contains the following restricted label(s): "
            f"{', '.join(label_intersect)}. Cannot assign to users."
        )
    else:
        # collect any mentioned (@username) users
        mentioned_users = re.findall(r"@[a-zA-Z0-9_-]+", comment_body)
        mentioned_users = [user[1:] for user in mentioned_users]
        mentioned_users.remove("aeon-actions-bot")
        # Assign commenter if comment includes "assign me"
        if "assign me" in body:
            mentioned_users.append(commenter)
        mentioned_users = set(mentioned_users)

        # Get permissions of the commenter
        commenter_permission = repo.get_collaborator_permission(commenter)

        for user in mentioned_users:
            if user != commenter and commenter_permission not in ["admin", "write"]:
                comment_msg = (
                    f"@{commenter}, you cannot assign @{user}"
                    " because you lack write access.\n"
                    "Only users with write access can assign others."
                )
                issue.create_comment(comment_msg)
                continue
            elif user != commenter and commenter_permission in ["admin", "write"]:
                issue.add_to_assignees(user)
                continue

            user_obj = g.get_user(user)
            permission = repo.get_collaborator_permission(user_obj)

            if permission in ["admin", "write"]:
                issue.add_to_assignees(user)
            else:
                # First check if the user is already assigned to this issue
                if user in [assignee.login for assignee in issue.assignees]:
                    continue

                # search for open issues only
                query = f"repo:{repo.full_name} is:issue is:open assignee:{user}"
                issues_assigned_to_user = g.search_issues(query)
                assigned_count = issues_assigned_to_user.totalCount

                if assigned_count >= 3:
                    # link to issue
                    assigned_issues_list = [
                        f"[#{assigned_issue.number}]({assigned_issue.html_url})"
                        for assigned_issue in issues_assigned_to_user
                    ]

                    comment_message = (
                        f"@{user}, already has {assigned_count} open issues assigned."
                        "Users without write access are limited to self-assigning "
                        "three issues.\n\n"
                        "Here are the open issues assigned:\n"
                        + "\n".join(
                            f"- {issue_link}" for issue_link in assigned_issues_list
                        )
                    )
                    issue.create_comment(comment_message)
                else:
                    issue.add_to_assignees(user)
