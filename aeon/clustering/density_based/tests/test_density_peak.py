"""Test for the DensityPeakClusterer module."""

import numpy as np
import pytest

from aeon.clustering.density_based._density_peak import DensityPeakClusterer

# spiral
DATA_STR = """
1.85	27.8	1
1.35	26.65	1
1.4	23.25	2
0.85	23.05	2
0.5	22.35	2
0.65	21.35	2
1.1	22.05	2
1.35	22.65	2
1.95	22.8	2
2.4	22.45	2
1.8	22	2
2.5	21.85	2
2.95	21.4	2
1.9	21.25	2
1.35	21.45	2
1.35	20.9	2
1.25	20.35	2
1.75	20.05	2
2	20.6	2
2.5	21	2
1.7	19.05	2
2.4	20.05	2
3.05	20.45	2
3.7	20.45	2
3.45	19.9	2
2.95	19.5	2
2.4	19.4	2
2.4	18.25	2
2.85	18.75	2
3.25	19.05	2
3.95	19.6	2
2.7	17.8	2
3.45	18.05	2
3.8	18.55	2
4	19.1	2
4.45	19.9	2
4.65	19.15	2
4.85	18.45	2
4.3	18.05	2
3.35	17.3	2
3.7	16.3	2
4.4	16.95	2
4.25	17.4	2
4.8	17.65	2
5.25	18.25	2
5.75	18.55	2
5.3	19.25	2
6.05	19.55	2
6.5	18.9	2
6.05	18.2	2
5.6	17.8	2
5.45	17.15	2
5.05	16.55	2
4.55	16.05	2
4.95	15.45	2
5.85	14.8	2
5.6	15.3	2
5.65	16	2
5.95	16.8	2
6.25	16.4	2
6.1	17.45	2
6.6	17.65	2
6.65	18.3	2
7.3	18.35	2
7.85	18.3	2
7.15	17.8	2
7.6	17.7	2
6.7	17.25	2
7.3	17.25	2
6.7	16.8	2
7.3	16.65	2
6.75	16.3	2
7.4	16.2	2
6.55	15.75	2
7.35	15.8	2
6.8	14.95	2
7.45	15.1	2
6.85	14.45	2
7.6	14.6	2
8.55	14.65	2
8.2	15.5	2
7.9	16.1	2
8.05	16.5	2
7.8	17	2
8	17.45	2
8.4	18.1	2
8.65	17.75	2
8.9	17.1	2
8.4	17.1	2
8.65	16.65	2
8.45	16.05	2
8.85	15.35	2
9.6	15.3	2
9.15	16	2
10.2	16	2
9.5	16.65	2
10.75	16.6	2
10.45	17.2	2
9.85	17.1	2
9.4	17.6	2
10.15	17.7	2
9.85	18.15	2
9.05	18.25	2
9.3	18.7	2
9.15	19.15	2
8.5	18.8	2
11.65	17.45	2
11.1	17.65	2
10.4	18.25	2
10	18.95	2
11.95	18.25	2
11.25	18.4	2
10.6	18.9	2
11.15	19	2
11.9	18.85	2
12.6	18.9	2
11.8	19.45	2
11.05	19.45	2
10.3	19.4	2
9.9	19.75	2
10.45	20	2
13.05	19.9	2
12.5	19.75	2
11.9	20.05	2
11.2	20.25	2
10.85	20.85	2
11.4	21.25	2
11.7	20.6	2
12.3	20.45	2
12.95	20.55	2
12.55	20.95	2
12.05	21.25	2
11.75	22.1	2
12.25	21.85	2
12.8	21.5	2
13.55	21	2
13.6	21.6	2
12.95	22	2
12.5	22.25	2
12.2	22.85	2
12.7	23.35	2
13	22.7	2
13.55	22.2	2
14.05	22.25	2
14.2	23.05	2
14.1	23.6	2
13.5	22.8	2
13.35	23.5	2
13.3	24	2
7.3	19.15	2
7.95	19.35	2
7.7	20.05	2
6.75	19.9	2
5.25	20.35	2
6.15	20.7	1
7	20.7	1
7.6	21.2	1
8.55	20.6	1
9.35	20.5	1
8.3	21.45	1
7.9	21.6	1
7.15	21.75	1
6.7	21.3	1
5.2	21.1	2
6.2	21.95	1
6.75	22.4	1
6.15	22.5	1
5.65	22.2	1
4.65	22.55	1
4.1	23.45	1
5.35	22.8	1
7.4	22.6	1
7.75	22.1	1
8.5	22.3	1
9.3	22	1
9.7	22.95	1
8.8	22.95	1
8.05	22.9	1
7.6	23.15	1
6.85	23	1
6.2	23.25	1
5.7	23.4	1
5.1	23.55	1
4.55	24.15	1
5.5	24	1
6.1	24.05	1
6.5	23.6	1
6.75	23.95	1
7.3	23.75	1
8.3	23.4	1
8.9	23.7	1
9.55	23.65	1
10.35	24.1	1
7.95	24.05	1
3.95	24.4	1
3.75	25.25	1
3.9	25.95	1
4.55	26.65	1
5.25	26.75	1
6.5	27.6	1
7.45	27.6	1
8.35	27.35	1
9.25	27.2	1
9.95	26.5	1
10.55	25.6	1
9.9	24.95	1
9.2	24.5	1
8.55	24.2	1
8.8	24.8	1
9.2	25.35	1
9.55	26.05	1
9.05	26.6	1
8.8	25.8	1
8.15	26.35	1
8.05	25.8	1
8.35	25.2	1
7.9	25.3	1
8.05	24.7	1
7.3	24.4	1
7.55	24.85	1
6.85	24.45	1
6.25	24.65	1
5.55	24.5	1
4.65	25.1	1
5	25.55	1
5.55	26.1	1
5.55	25.25	1
6.2	25.2	1
6.8	25.05	1
7.4	25.25	1
6.65	25.45	1
6.15	25.8	1
6.5	26.1	1
6.6	26.6	1
7.7	26.65	1
7.5	26.2	1
7.5	25.65	1
7.05	25.85	1
6.9	27.15	1
6.15	26.9	1

"""


def load_dataset():
    """
    Load the provided dataset from the multiline string.

    Each row has three columns (x, y, and an unused label).
    Returns the x and y coordinates as (n_samples, 2).
    """
    lines = DATA_STR.strip().splitlines()
    data_list = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            x = float(parts[0])
            y = float(parts[1])
            data_list.append([x, y])
    return np.array(data_list)


@pytest.fixture
def dataset():
    """Load the dataset."""
    return load_dataset()


def test_density_peak_clusterer(dataset):
    """Test the DensityPeakClusterer with the provided dataset."""
    clusterer = DensityPeakClusterer(
        gauss_cutoff=True,
        cutoff_distance="auto",
        distance_metric="euclidean",
        density_threshold=8,
        distance_threshold=5,
        anormal=False,
    )

    clusterer.fit(dataset)

    print("Cluster Labels:")  # noqa
    print(clusterer.labels_)  # noqa
    print("\nCluster Centers:")  # noqa
    print(clusterer.cluster_centers)  # noqa

    assert clusterer.labels_ is not None, "Cluster labels should not be None"
    assert len(clusterer.labels_) == len(
        dataset
    ), "Number of labels should match number of data points"
    assert clusterer.cluster_centers is not None, "Cluster centers should not be None"
    assert (
        len(clusterer.cluster_centers) > 0
    ), "There should be at least one cluster center"

    clusterer.plot(mode="all", title="Density Peak Clustering")


# use pytest -s
