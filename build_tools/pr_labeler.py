"""Labels PRs based on title.

Must be run in a github action with the pull_request_target event.

Based on the scikit-learn v1.3.1 label_title_regex.py script.
"""

import json
import os
import re

from github import Github

context_dict = json.loads(os.getenv("CONTEXT_GITHUB"))

repo = context_dict["repository"]
g = Github(context_dict["token"])
repo = g.get_repo(repo)
pr_number = context_dict["event"]["number"]
pr = repo.get_pull(number=pr_number)
labels = [label.name for label in pr.get_labels()]

print(list(labels))  # noqa: T201

# title labels
title = pr.title

title_regex_to_labels = [
    (r"\bENH\b", "enhancement"),
    (r"\bMNT\b", "maintenance"),
    (r"\bBUG\b", "bug"),
    (r"\bDOC\b", "documentation"),
    (r"\bGOV\b", "governance"),
]

title_labels_to_add = [
    label for regex, label in title_regex_to_labels if re.search(regex, title)
]

print(title_labels_to_add)  # noqa: T201

# content labels
paths = [file.filename for file in pr.get_files()]

print(paths)  # noqa: T201

content_paths_to_labels = [
    ("aeon/anomaly_detection/", "anomaly detection"),
    ("aeon/benchmarking/", "benchmarking"),
    ("aeon/classification*", "classification"),
    ("aeon/clustering/", "clustering"),
    ("aeon/datasets/", "datasets"),
    ("aeon/datatypes/", "datatypes"),
    ("aeon/distances/", "distances"),
    ("examples/", "examples"),
    ("aeon/forecasting/", "forecasting"),
    ("aeon/networks/", "networks"),
    ("aeon/regression/", "regression"),
    ("aeon/segmentation/", "segmentation"),
    ("aeon/similarity_search/", "similarity search"),
    ("aeon/transformations/", "transformation"),
]

content_labels_to_add = [
    label
    for package, label in content_paths_to_labels
    if any([package in path for path in paths])
]
content_labels_to_add = list(set(content_labels_to_add))
content_labels_to_add = [] if len(content_labels_to_add) > 3 else content_labels_to_add

print(content_labels_to_add)  # noqa: T201

# add to PR
if title_labels_to_add or content_labels_to_add:
    pr.add_to_labels(*title_labels_to_add + content_labels_to_add)

with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
    print(f"title-labels-new={title_labels_to_add}", file=fh)  # noqa: T201
    print(f"content-labels-new={content_labels_to_add}", file=fh)  # noqa: T201
