from aeon.clustering import TimeSeriesKMeans
from aeon.datasets import load_gunpoint

if __name__ == "__main__":

    X, y = load_gunpoint(split="train")

    temp = ["soft_msm", "soft_dtw", "soft_twe"]
    for dist in temp:
        clst = TimeSeriesKMeans(
            n_clusters=2,
            init="kmeans++",
            distance=dist,
            n_init=1,
            max_iter=300,
            random_state=1,
            averaging_method="soft_ba",
            distance_params={"gamma": 0.01},
            average_params={"gamma": 0.01},
        )

        preds = clst.fit_predict(X)
