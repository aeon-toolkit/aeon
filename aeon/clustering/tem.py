from aeon.clustering._kshape_compare import TimeSeriesKShapeCompare
from aeon.clustering._original_kshape_simple import TimeSeriesKShapeBare
# from aeon.clustering._k_shape import TimeSeriesKShape
from aeon.datasets import load_gunpoint
from sklearn.metrics import adjusted_rand_score

if __name__ == "__main__":
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")

    kshape = TimeSeriesKShapeCompare(n_clusters=2, n_init=1, random_state=1, version="original", centroid_init="kmeans++")
    kshape.fit_predict(X_train)

    print(kshape.labels_)
    print(kshape.predict(X_test))
    print(kshape.cluster_centers_.shape)
    print(adjusted_rand_score(y_train, kshape.labels_))

    kshape = TimeSeriesKShapeCompare(n_clusters=2, n_init=1, random_state=1, version="tslearn", centroid_init="kmeans++")
    kshape.fit_predict(X_train)

    print(kshape.labels_)
    print(kshape.predict(X_test))
    print(kshape.cluster_centers_.shape)
    print(adjusted_rand_score(y_train, kshape.labels_))

    kshape = TimeSeriesKShapeBare(n_clusters=3, random_state=1, centroid_init="random")
    kshape.fit_predict(X_train)

    print(kshape.labels_)
    print(kshape.predict(X_test))
    print(kshape.cluster_centers_.shape)
    print(adjusted_rand_score(y_train, kshape.labels_))
