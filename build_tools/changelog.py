# -*- coding: utf-8 -*-
"""Myst Markdown changelog generator."""

import os
from collections import defaultdict
from typing import Dict, List

import httpx
from dateutil import parser

HEADERS = {
    "Accept": "application/vnd.github.v3+json",
}

if os.getenv("GITHUB_TOKEN") is not None:
    HEADERS["Authorization"] = f"token {os.getenv('GITHUB_TOKEN')}"

OWNER = "aeon-toolkit"
REPO = "aeon"
GITHUB_REPOS = "https://api.github.com/repos"

def fetch_merged_pull_requests(page: int = 1) -> List[Dict]:  # noqa
    """Fetch a page of pull requests"""
    params = {
        "base": "main",
        "state": "closed",
        "page": page,
        "per_page": 50,
        "sort": "updated",
        "direction": "desc",
    }
    r = httpx.get(
        f"{GITHUB_REPOS}/{OWNER}/{REPO}/pulls",
        headers=HEADERS,
        params=params,
    )
    return [pr for pr in r.json() if pr["merged_at"]]


def fetch_latest_release():  # noqa
    response = httpx.get(
        f"{GITHUB_REPOS}/{OWNER}/{REPO}/releases/latest", headers=HEADERS
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(response.text, response.status_code)


def fetch_pull_requests_since_last_release() -> List[Dict]:  # noqa
    """Fetch pull requests and filter based on merged date"""
    release = fetch_latest_release()
    published_at = parser.parse(release["published_at"])
    print(  # noqa
        f"Latest release {release['tag_name']} was published at {published_at}"
    )

    is_exhausted = False
    page = 1
    all_pulls = []
    while not is_exhausted:
        pulls = fetch_merged_pull_requests(page=page)
        all_pulls.extend(
            [p for p in pulls if parser.parse(p["merged_at"]) > published_at]
        )
        is_exhausted = any(parser.parse(p["merged_at"]) < published_at for p in pulls) or len(pulls) == 0
        page += 1
    return all_pulls


def github_compare_tags(tag_left: str, tag_right: str = "HEAD"):  # noqa
    """Compare commit between two tags"""
    response = httpx.get(
        f"{GITHUB_REPOS}/{OWNER}/{REPO}/compare/{tag_left}...{tag_right}"
    )
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(response.text, response.status_code)


EXCLUDED_USERS = ["github-actions[bot]"]

def render_contributors(prs: List, fmt: str = "myst"):  # noqa
    """Find unique authors and print a list in  given format"""
    authors = sorted({pr["user"]["login"] for pr in prs}, key=lambda x: x.lower())

    header = "Contributors\n"
    if fmt == "github":
        print(f"### {header}")  # noqa
        print(", ".join(f"@{user}" for user in authors if user not in EXCLUDED_USERS))  # noqa
    elif fmt == "myst":
        print(f"## {header}")  # noqa
        print(",\n".join("{user}" + f"`{user}`" for user in authors if user not in EXCLUDED_USERS))  # noqa


def assign_prs(prs, categs: List[Dict[str, List[str]]]):  # noqa
    """Assign PR to categories based on labels"""
    assigned = defaultdict(list)

    for i, pr in enumerate(prs):
        for cat in categs:
            pr_labels = [label["name"] for label in pr["labels"]]
            if cat["title"] != "Not Included" and "no changelog" in pr_labels:
                continue
            if not set(cat["labels"]).isdisjoint(set(pr_labels)):
                assigned[cat["title"]].append(i)

    assigned["Other"] = list(
        set(range(len(prs))) - {i for _, l in assigned.items() for i in l}
    )

    if "Not Included" in assigned:
        assigned.pop("Not Included")

    return assigned


def render_row(pr):  # noqa
    """Render a single row with PR in Myst Markdown format"""
    print(  # noqa
        "-",
        pr["title"],
        "({pr}" + f"`{pr['number']}`)",
        "{user}" + f"`{pr['user']['login']}`",
    )


def render_changelog(prs, assigned):  # noqa
    # sourcery skip: use-named-expression
    """Render changelog"""
    for title, _ in assigned.items():
        pr_group = [prs[i] for i in assigned[title]]
        if pr_group:
            print(f"\n## {title}\n")  # noqa

            for pr in sorted(pr_group, key=lambda x: parser.parse(x["merged_at"])):
                render_row(pr)


if __name__ == "__main__":
    categories = [
        {"title": "Enhancements", "labels": ["enhancement"]},
        {"title": "Fixes", "labels": ["bug"]},
        {"title": "Maintenance", "labels": ["maintenance"]},
        {"title": "Refactored", "labels": ["refactor"]},
        {"title": "Documentation", "labels": ["documentation"]},
        {"title": "Not Included", "labels": ["no changelog"]},  # this is deleted
    ]

    pulls = fetch_pull_requests_since_last_release()
    print(f"Found {len(pulls)} merged PRs since last release")  # noqa
    assigned = assign_prs(pulls, categories)
    render_changelog(pulls, assigned)
    print()  # noqa
    render_contributors(pulls)

    release = fetch_latest_release()
    diff = github_compare_tags(release["tag_name"])
    if diff["total_commits"] != len(pulls):
        raise ValueError(
            "Something went wrong and not all PR were fetched. "
            f'There are {len(pulls)} PRs but {diff["total_commits"]} in the diff. '
            "Please verify that all PRs are included in the changelog."
        )  # noqa
