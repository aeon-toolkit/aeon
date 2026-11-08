"""Test that the Zenodo record IDs used by the dataset loaders are up to date."""

import json
import time
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from aeon.datasets._data_loaders import CONNECTION_ERRORS
from aeon.datasets.dataset_collections import tsml_archives
from aeon.datasets.tsc_datasets import tsc_zenodo
from aeon.datasets.tser_datasets import tsr_zenodo
from aeon.testing.testing_config import PR_TESTING

# Dictionaries mapping names to Zenodo record IDs in the tsml community.
ZENODO_ID_DICTS = {
    "tsc_zenodo": tsc_zenodo,
    "tsr_zenodo": tsr_zenodo,
    "tsml_archives": tsml_archives,
}

# Entries deliberately kept on an older version of their Zenodo record, as
# {dict name: {entry name: reason}}. A pin is reported if the entry is no longer in
# the dict or its ID has become the latest version, so pins do not go stale.
PINNED_OLD_VERSIONS = {
    "tsc_zenodo": {},
    "tsr_zenodo": {},
    "tsml_archives": {},
}


def _get_json(url, retries=3):
    """Return the JSON at url, waiting and retrying if rate limited."""
    for attempt in range(retries + 1):
        try:
            with urlopen(url, timeout=60) as response:
                return json.load(response)
        except HTTPError as e:
            if e.code != 429 or attempt == retries:
                raise
            time.sleep(int(e.headers.get("Retry-After", 60)))


def _latest_community_versions():
    """Return {concept record ID: latest record ID} for the tsml community."""
    latest = {}
    # 25 is the largest page size Zenodo allows without authentication
    url = "https://zenodo.org/api/communities/tsml/records?size=25&sort=oldest"
    while url:
        page = _get_json(url)
        for record in page["hits"]["hits"]:
            latest[int(record["conceptrecid"])] = int(record["id"])
        url = page["links"].get("next")
    return latest


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because it reads from Zenodo.",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_zenodo_ids_are_latest_versions():
    """Test stored Zenodo IDs point at the latest version of each record.

    Publishing a new version of a dataset on Zenodo creates a new record ID, while
    the old ID keeps serving the old files. Entries that should stay on an old
    version must be listed in PINNED_OLD_VERSIONS.
    """
    latest = _latest_community_versions()
    latest_ids = set(latest.values())

    problems = []
    for dict_name, ids in ZENODO_ID_DICTS.items():
        pinned = PINNED_OLD_VERSIONS.get(dict_name, {})
        for name in pinned:
            if name not in ids:
                problems.append(f"{dict_name}[{name!r}] is pinned but not in the dict.")

        for name, record_id in ids.items():
            if record_id in latest_ids:
                if name in pinned:
                    problems.append(
                        f"{dict_name}[{name!r}] is pinned but {record_id} is the "
                        f"latest version, remove it from PINNED_OLD_VERSIONS."
                    )
                continue
            if name in pinned:
                continue

            try:
                record = _get_json(f"https://zenodo.org/api/records/{record_id}")
            except HTTPError as e:
                if e.code not in (404, 410):
                    raise
                problems.append(
                    f"{dict_name}[{name!r}] = {record_id} was not found on Zenodo "
                    f"(HTTP {e.code})."
                )
                continue

            concept_id = int(record["conceptrecid"])
            if concept_id in latest:
                problems.append(
                    f"{dict_name}[{name!r}] = {record_id}, but the latest version "
                    f"is {latest[concept_id]}."
                )
            else:
                problems.append(
                    f"{dict_name}[{name!r}] = {record_id} is not in the Zenodo "
                    "tsml community."
                )

    assert not problems, "Out of date Zenodo IDs:\n" + "\n".join(problems)
