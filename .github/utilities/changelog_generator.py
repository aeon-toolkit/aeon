"""Myst Markdown changelog generator."""

import sys
from collections import OrderedDict

import httpx
from dateutil import parser

HEADERS = {
    "Accept": "application/vnd.github.v3+json",
}

OWNER = "aeon-toolkit"
REPO = "aeon"
GITHUB_REPOS = "https://api.github.com/repos"
EXCLUDED_USERS = [
    "github-actions[bot]",
    "allcontributors[bot]",
    "sweep-ai[bot]",
    "aeon-actions-bot[bot]",
    "dependabot[bot]",
]


def fetch_merged_pull_requests(page: int = 1) -> list[dict]:
    """Fetch a page of pull requests."""
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


def fetch_latest_release() -> dict:
    """Fetch latest release."""
    response = httpx.get(
        f"{GITHUB_REPOS}/{OWNER}/{REPO}/releases/latest", headers=HEADERS
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(response.text, response.status_code)


def fetch_pull_requests_since_last_release() -> list[dict]:
    """Fetch pull requests and filter based on merged date."""
    release = fetch_latest_release()
    published_at = parser.parse(release["published_at"])
    print(  # noqa: T201
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
        is_exhausted = (
            any(parser.parse(p["merged_at"]) < published_at for p in pulls)
            or len(pulls) == 0
        )
        page += 1
    return all_pulls


def github_compare_tags(tag_left: str, tag_right: str = "HEAD") -> dict:
    """Compare commit between two tags."""
    response = httpx.get(
        f"{GITHUB_REPOS}/{OWNER}/{REPO}/compare/{tag_left}...{tag_right}"
    )
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(response.text, response.status_code)


def render_contributors(prs: list, fmt: str = "myst", n_prs: int = -1):
    """Find unique authors and print a list in  given format."""
    authors = sorted({pr["user"]["login"] for pr in prs}, key=lambda x: x.lower())

    header = "Contributors\n"
    if fmt == "github":
        print(f"### {header}")  # noqa: T201
        print(  # noqa: T201
            ", ".join(f"@{user}" for user in authors if user not in EXCLUDED_USERS)
        )
    elif fmt == "myst":
        print(f"## {header}")  # noqa: T201
        print(  # noqa: T201
            "The following have contributed to this release through a collective "
            f"{n_prs} GitHub Pull Requests:\n"
        )
        print(  # noqa: T201
            ",\n".join(
                "{user}" + f"`{user}`" for user in authors if user not in EXCLUDED_USERS
            )
        )


def assign_pr_category(
    assigned: dict, categories: list[list], pr_idx: int, pr_labels: list, pkg_title: str
):
    """Assign a PR to a category."""
    has_category = False
    for cat in categories:
        if not set(cat[1]).isdisjoint(set(pr_labels)):
            has_category = True

            if cat[0] not in assigned[pkg_title]:
                assigned[pkg_title][cat[0]] = []

            assigned[pkg_title][cat[0]].append(pr_idx)

    if not has_category:
        if "Other" not in assigned[pkg_title]:
            assigned[pkg_title]["Other"] = []

        assigned[pkg_title]["Other"].append(pr_idx)


def assign_prs(
    prs: list[dict], packages: list[list], categories: list[list]
) -> tuple[dict, int]:
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
            if not set(pkg[1]).isdisjoint(set(pr_labels)):
                has_package = True

                if pkg[0] not in assigned:
                    assigned[pkg[0]] = {}

                assign_pr_category(assigned, categories, i, pr_labels, pkg[0])

        if not has_package:
            if "Other" not in assigned:
                assigned["Other"] = OrderedDict()

            assign_pr_category(assigned, categories, i, pr_labels, "Other")

    # order assignments
    assigned = OrderedDict({k: v for k, v in sorted(assigned.items())})
    if "Other" in assigned:
        assigned.move_to_end("Other")

    for key in assigned:
        assigned[key] = OrderedDict({k: v for k, v in sorted(assigned[key].items())})
        if "Other" in assigned[key]:
            assigned[key].move_to_end("Other")

    return assigned, prs_removed


def render_row(pr: dict):  # noqa
    """Render a single row with PR in Myst Markdown format."""
    print(  # noqa: T201
        "-",
        pr["title"],
        "({pr}" + f"`{pr['number']}`)",
        "{user}" + f"`{pr['user']['login']}`",
    )


def render_changelog(prs: list[dict], assigned: dict):
    """Render changelog."""
    for pkg_title, group in assigned.items():
        print(f"\n## {pkg_title}")  # noqa: T201

        for cat_title, pr_idx in group.items():
            print(f"\n### {cat_title}\n")  # noqa: T201
            pr_group = [prs[i] for i in pr_idx]

            for pr in sorted(pr_group, key=lambda x: parser.parse(x["merged_at"])):
                render_row(pr)


if __name__ == "__main__":
    print("access token:", file=sys.stderr)  # noqa: T201
    token = input()

    if token is not None and token != "":
        HEADERS["Authorization"] = f"token {token}"

    # if you edit these, consider editing the PR template as well
    packages = [
        ["Anomaly Detection", ["anomaly detection"]],
        ["Benchmarking", ["benchmarking"]],
        ["Classification", ["classification"]],
        ["Clustering", ["clustering"]],
        ["Datasets", ["datasets"]],
        ["Distances", ["distances"]],
        ["Forecasting", ["forecasting"]],
        ["Networks", ["networks"]],
        ["Regression", ["regression"]],
        ["Segmentation", ["segmentation"]],
        ["Similarity Search", ["similarity search"]],
        ["Unit Testing", ["testing"]],
        ["Transformations", ["transformations"]],
        ["Visualisations", ["visualisation"]],
    ]
    categories = [
        ["Bug Fixes", ["bug"]],
        ["Documentation", ["documentation"]],
        ["Enhancements", ["enhancement"]],
        ["Maintenance", ["maintenance"]],
        ["Refactored", ["refactor"]],
        ["Deprecation", ["deprecation"]],
    ]

    pulls = fetch_pull_requests_since_last_release()
    print(f"Found {len(pulls)} merged PRs since last release")  # noqa: T201

    assigned, prs_removed = assign_prs(pulls, packages, categories)

    render_changelog(pulls, assigned)
    print()  # noqa: T201
    render_contributors(pulls, fmt="myst", n_prs=len(pulls) - prs_removed)

    release = fetch_latest_release()
    diff = github_compare_tags(release["tag_name"])
    if diff["total_commits"] != len(pulls):
        raise ValueError(
            "Something went wrong and not all PR were fetched. "
            f"There are {len(pulls)} PRs but {diff['total_commits']} in the diff. "
            "Please verify that all PRs are included in the changelog."
        )
