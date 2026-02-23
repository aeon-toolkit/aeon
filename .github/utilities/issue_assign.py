"""Script for the GitHub issue self-assign bot.

Checks if a comment on an issue or PR includes the trigger phrase and a mentioned user.
If it does, it assigns or unassigns the issue to the mentioned user if they have
permissions.
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
commenter_permission = repo.get_collaborator_permission(commenter)
has_write_permission = commenter_permission in ["admin", "write"]

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

        access_error = False
        for user in mentioned_users:
            # Can only assign others if the commenter has write access
            if user != commenter and not has_write_permission:
                if not access_error:
                    issue.create_comment(
                        "Cannot assign other users to issues without write access."
                    )
                    access_error = True
                continue

            # If the user is already assigned to this issue, remove them
            if user in [assignee.login for assignee in issue.assignees]:
                issue.remove_from_assignees(user)
                continue

            # If the commenter has write access, just assign
            if has_write_permission:
                issue.add_to_assignees(user)
            else:
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

                    issue.create_comment(
                        f"@{user}, already has {assigned_count} open issues assigned."
                        "Users without write access are limited to self-assigning "
                        "three issues.\n\n"
                        "Here are the open issues assigned:\n"
                        + "\n".join(
                            f"- {issue_link}" for issue_link in assigned_issues_list
                        )
                    )
                else:
                    issue.add_to_assignees(user)
