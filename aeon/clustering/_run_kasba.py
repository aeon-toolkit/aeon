import numpy as np
from sklearn.metrics import adjusted_rand_score
from statsmodels.tsa.statespace.simulation_smoother import check_random_state
from sympy.core.sympify import kernS

from aeon.clustering import KASBA, KESBA
from aeon.datasets import load_acsf1, load_gunpoint
from aeon.distances import pairwise_distance


def _first_init(X, n_clusters):
    cluster_centres = X[0:n_clusters]
    pw_dists = pairwise_distance(
        X,
        cluster_centres,
        metric="msm",
        **{"c": 1.0},
    )
    min_dists = pw_dists.min(axis=1)
    labels = pw_dists.argmin(axis=1)
    return cluster_centres, min_dists, labels


if __name__ == "__main__":
    # X_train, y_train = load_gunpoint(split="train")
    X_train, y_train = load_acsf1(split="train")
    n_clusters = len(set(list(y_train)))
    verbose = True

    kasba_clust = KESBA(
        n_clusters=n_clusters,
        random_state=1,
        skip_barycentre_if_labels_no_change=True,
        use_new_kmeans_plus=True,
        use_check_centres_change_assignment=True,
    )

    kesba_clust = KESBA(
        n_clusters=n_clusters,
        random_state=1,
    )

    print("=========== KASBA =========")
    kasba_labels = kasba_clust.fit_predict(X_train)
    print("=========== KESBA =========")
    kesba_labels = kesba_clust.fit_predict(X_train)

    # print(f"Init equal {np.array_equal(kasba_clust.init[0], kesba_clust.init[0])}")
    # print(f"Init dists equal {np.array_equal(kasba_clust.init[1], kesba_clust.init[1])}")
    # print(f"Init labels equal {np.array_equal(kasba_clust.init[2], kesba_clust.init[2])}")

    print("KESBA ARI: ", adjusted_rand_score(y_train, kesba_labels))
    print("KASBA ARI: ", adjusted_rand_score(y_train, kasba_labels))

    print(f"Labels match {np.array_equal(kesba_labels, kasba_labels)}")