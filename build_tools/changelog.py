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

if os.getenv("GITHUB_TOKEN") is not None and os.getenv("GITHUB_TOKEN") != "":
    HEADERS["Authorization"] = f"token {os.getenv('GITHUB_TOKEN')}"

OWNER = "aeon-toolkit"
REPO = "aeon"
GITHUB_REPOS = "https://api.github.com/repos"
EXCLUDED_USERS = ["github-actions[bot]"]

def fetch_merged_pull_requests(page: int = 1) -> List[Dict]:
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


def fetch_latest_release() -> Dict:
    """Fetch latest release."""
    response = httpx.get(
        f"{GITHUB_REPOS}/{OWNER}/{REPO}/releases/latest", headers=HEADERS
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(response.text, response.status_code)


def fetch_pull_requests_since_last_release() -> List[Dict]:
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


def github_compare_tags(tag_left: str, tag_right: str = "HEAD") -> Dict:
    """Compare commit between two tags"""
    response = httpx.get(
        f"{GITHUB_REPOS}/{OWNER}/{REPO}/compare/{tag_left}...{tag_right}"
    )
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(response.text, response.status_code)


def render_contributors(prs: list, fmt: str = "myst", n_prs: int = -1):
    """Find unique authors and print a list in  given format"""
    authors = sorted({pr["user"]["login"] for pr in prs}, key=lambda x: x.lower())

    header = "Contributors\n"
    if fmt == "github":
        print(f"### {header}")  # noqa
        print(", ".join(f"@{user}" for user in authors if user not in EXCLUDED_USERS))  # noqa
    elif fmt == "myst":
        print(f"## {header}")  # noqa
        print(f"The following have contributed to this release through a collective {n_prs} GitHub Pull Requests:\n")
        print(",\n".join("{user}" + f"`{user}`" for user in authors if user not in EXCLUDED_USERS))  # noqa


def assign_pr_category(assigned: Dict, categories: List[Dict], pr_idx: int, pr_labels: List, pkg_title: str):
    """Assign a PR to a category."""
    has_category = False
    for cat in categories:
        if not set(cat["labels"]).isdisjoint(set(pr_labels)):
            has_category = True

            if cat["title"] not in assigned[pkg_title]:
                assigned[pkg_title][cat["title"]] = []

            assigned[pkg_title][cat["title"]].append(pr_idx)

    if not has_category:
        if "Other" not in assigned[pkg_title]:
            assigned[pkg_title]["Other"] = []

        assigned[pkg_title]["Other"].append(pr_idx)


def assign_prs(prs: List[Dict], packages: List[Dict], categories: List[Dict]) -> Dict:
    """Assign all PRs to packages and categories based on labels."""
    assigned = {}
    prs_removed = 0

    for i, pr in enumerate(prs):
        pr_labels = [label["name"] for label in pr["labels"]]

        if "no changelog" in pr_labels:
            prs_removed += 1
            continue

        has_package = False
        for pkg in packages:
            if not set(pkg["labels"]).isdisjoint(set(pr_labels)):
                has_package = True

                if pkg["title"] not in assigned:
                    assigned[pkg["title"]] = {}

                assign_pr_category(assigned, categories, i, pr_labels, pkg["title"])

        if not has_package:
            if "Other" not in assigned:
                assigned["Other"] = {}

            assign_pr_category(assigned, categories, i, pr_labels, "Other")

    return assigned, prs_removed


def render_row(pr: Dict):  # noqa
    """Render a single row with PR in Myst Markdown format"""
    print(  # noqa
        "-",
        pr["title"],
        "({pr}" + f"`{pr['number']}`)",
        "{user}" + f"`{pr['user']['login']}`",
    )


def render_changelog(prs: List[Dict], assigned: Dict):
    """Render changelog"""
    for pkg_title, group in assigned.items():
        print(f"\n## {pkg_title}")  # noqa

        for cat_title, pr_idx in group.items():
            print(f"\n### {cat_title}\n")  # noqa
            pr_group = [prs[i] for i in pr_idx]

            for pr in sorted(pr_group, key=lambda x: parser.parse(x["merged_at"])):
                render_row(pr)


if __name__ == "__main__":
    # don't commit the actual token, it will get revoked!
    os.environ["GITHUB_TOKEN"] = ""

    # if you edit these, consider editing the PR template as well
    packages = [
        {"title": "Annotation", "labels": ["annotation"]},
        {"title": "Benchmarking", "labels": ["benchmarking"]},
        {"title": "Classification", "labels": ["classification"]},
        {"title": "Clustering", "labels": ["clustering"]},
        {"title": "Distances", "labels": ["distances"]},
        {"title": "Forecasting", "labels": ["forecasting"]},
        {"title": "Regression", "labels": ["regression"]},
        {"title": "Transformations", "labels": ["transformations"]},
    ]
    categories = [
        {"title": "Bug Fixes", "labels": ["bug"]},
        {"title": "Documentation", "labels": ["documentation"]},
        {"title": "Enhancements", "labels": ["enhancement"]},
        {"title": "Maintenance", "labels": ["maintenance"]},
        {"title": "Refactored", "labels": ["refactor"]},
    ]

    pulls = fetch_pull_requests_since_last_release()
    print(f"Found {len(pulls)} merged PRs since last release")  # noqa

    assigned, prs_removed = assign_prs(pulls, packages, categories)

    render_changelog(pulls, assigned)
    print()  # noqa
    render_contributors(pulls, fmt="myst", n_prs=len(pulls) - prs_removed)

    release = fetch_latest_release()
    diff = github_compare_tags(release["tag_name"])
    if diff["total_commits"] != len(pulls):
        raise ValueError(
            "Something went wrong and not all PR were fetched. "
            f"There are {len(pulls)} PRs but {diff['total_commits']} in the diff. "
            "Please verify that all PRs are included in the changelog."
        )
