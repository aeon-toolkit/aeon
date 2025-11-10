from aeon.clustering._original_kshape import TimeSeriesKShape
from aeon.datasets import load_gunpoint

if __name__ == "__main__":
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")

    kshape = TimeSeriesKShape(n_clusters=3)
    kshape.fit_predict(X_train)

    print(kshape.labels_)

    print(kshape.predict(X_test))

    print(kshape.cluster_centers_.shape)
