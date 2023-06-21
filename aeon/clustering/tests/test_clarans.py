from aeon.clustering.clarans import TimeSeriesCLARANS
from aeon.datasets import load_gunpoint
from sklearn import metrics

def test_clarans():
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")

    clara = TimeSeriesCLARANS(
        random_state=1,
        n_init=2,
        init_algorithm="first",
        distance="euclidean",
        n_clusters=2,
    )
    train_medoids_result = clara.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_medoids_result)
    test_medoids_result = clara.predict(X_test)
    test_score = metrics.rand_score(y_test, test_medoids_result)
    proba = clara.predict_proba(X_test)
