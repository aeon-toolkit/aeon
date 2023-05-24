# -*- coding: utf-8 -*-
"""Classifier Experiments: code to run experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard format.
"""

__author__ = ["TonyBagnall"]

import os

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

import sys
import time
from datetime import datetime

import numba
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import davies_bouldin_score
from aeon.clustering.k_means import TimeSeriesKMeans
from aeon.clustering.k_medoids import TimeSeriesKMedoids
from aeon.clustering.partitioning import TimeSeriesLloyds
from aeon.datasets import load_from_tsfile as load_ts
from aeon.datasets import write_results_to_uea_format
from aeon.utils.sampling import stratified_resample


def run_clustering_experiment(
    trainX,
    clusterer,
    results_path,
    trainY=None,
    testX=None,
    testY=None,
    cls_name=None,
    dataset_name=None,
    resample_id=0,
    overwrite=True,
):
    """
    Run a clustering experiment and save the results to file.

    Method to run a basic experiment and write the results to files called
    testFold<resampleID>.csv and, if required, trainFold<resampleID>.csv. This
    version loads the data from file based on a path. The clusterer is always trained on
    the required input data trainX. Output to trainResample<resampleID>.csv will be
    the predicted clusters of trainX. If trainY is also passed, these are written to
    file. If the clusterer makes probabilistic predictions, these are also written to
    file. See write_results_to_uea_format for more on the output. Be warned,
    this method will always overwrite existing results, check bvefore calling or use
    load_and_run_clustering_experiment instead.

    Parameters
    ----------
    trainX : pd.DataFrame or np.array
        The data to cluster.
    clusterer : BaseClusterer
        The clustering object
    results_path : str
        Where to write the results to
    trainY : np.array, default = None
        Train data tue class labels, only used for file writing, ignored by the
        clusterer
    testX : pd.DataFrame or np.array, default = None
        Test attribute data, if present it is used for predicting testY
    testY : np.array, default = None
        Test data true class labels, only used for file writing, ignored by the
        clusterer
    cls_name : str, default = None
        Name of the clusterer, written to the results file, ignored if None
    dataset_name : str, default = None
        Name of problem, written to the results file, ignored if None
    resample_id : int, default = 0
        Resample identifier, defaults to 0

    """
    if not overwrite:
        full_path = (
            f"{results_path}/{cls_name}/Predictions/"
            f"{dataset_name}/testResample{resample_id}.csv"
        )
        if os.path.exists(full_path):
            return

    # Build the clusterer on train data, recording how long it takes
    le = preprocessing.LabelEncoder()
    le.fit(trainY)
    trainY = le.transform(trainY)
    testY = le.transform(testY)
    start = int(round(time.time() * 1000))
    clusterer.fit(trainX)
    build_time = int(round(time.time() * 1000)) - start
    train_preds = clusterer.predict(trainX)
    train_probs = clusterer.predict_proba(trainX)

    start = int(round(time.time() * 1000))
    test_preds = clusterer.predict(testX)
    test_time = int(round(time.time() * 1000)) - start
    test_probs = clusterer.predict_proba(testX)
    second = str(clusterer.get_params())
    if isinstance(clusterer, TimeSeriesLloyds):
        second = f"{second},inertia,{clusterer.inertia_},n_its,{clusterer.n_iter_}"
    second = second.replace("\n", " ")
    second = second.replace("\r", " ")
    third = (
        f"0,{build_time},{test_time},-1,-1,{len(np.unique(trainY))},"
        f"{len(np.unique(trainY))}"
    )
    write_results_to_uea_format(
        first_line_comment="Generated by clustering_experiments on "
        + str(datetime.today()),
        second_line=second,
        third_line=third,
        output_path=results_path,
        estimator_name=cls_name,
        resample_seed=resample_id,
        y_pred=test_preds,
        predicted_probs=test_probs,
        dataset_name=dataset_name,
        y_true=testY,
        split="TEST",
        full_path=False,
        timing_type="MILLISECONDS",
    )

    second = str(clusterer.get_params())
    if isinstance(clusterer, TimeSeriesLloyds):
        second = f"{second},inertia,{clusterer.inertia_},n_its,{clusterer.n_iter_}"
    second = second.replace("\n", " ")
    second = second.replace("\r", " ")
    third = (
        f"0,{build_time},{test_time},-1,-1,{len(np.unique(trainY))},"
        f"{len(np.unique(trainY))}"
    )
    write_results_to_uea_format(
        first_line_comment="Generated by clustering_experiments on "
        + str(datetime.today()),
        second_line=second,
        third_line=third,
        output_path=results_path,
        estimator_name=cls_name,
        resample_seed=resample_id,
        y_pred=train_preds,
        predicted_probs=train_probs,
        dataset_name=dataset_name,
        y_true=trainY,
        split="TRAIN",
        full_path=False,
        timing_type="MILLISECONDS",
    )


def load_and_run_clustering_experiment(
    problem_path,
    results_path,
    dataset,
    clusterer,
    resample_id=0,
    cls_name=None,
    overwrite=False,
    format=".ts",
    train_file=False,
):
    """Run a clustering experiment.

    Method to run a basic experiment and write the results to files called
    testFold<resampleID>.csv and, if required, trainFold<resampleID>.csv. This
    version loads the data from file based on a path. The
    clusterer is always trained on the

    Parameters
    ----------
    problem_path : str
        Location of problem files, full path.
    results_path : str
        Location of where to write results. Any required directories will be created
    dataset : str
        Name of problem. Files must be  <problem_path>/<dataset>/<dataset>+
        "_TRAIN"+format, same for "_TEST"
    clusterer : the clusterer
    cls_name : str, default =None
        determines what to call the write directory. If None, it is set to
        type(clusterer).__name__
    resample_id : int, default = 0
        Seed for resampling. If set to 0, the default train/test split from file is
        used. Also used in output file name.
    overwrite : boolean, default = False
        if False, this will only build results if there is not a result file already
        present. If True, it will overwrite anything already there.
    format: string, default = ".ts"
        Valid formats are ".ts", ".arff", ".tsv" and ".long". For more info on
        format, see   examples/loading_data.ipynb
    train_file: boolean, default = False
        whether to generate train files or not. If true, it performs a 10xCV on the
        train and saves
    """
    if cls_name is None:
        cls_name = type(clusterer).__name__

    # Set up the file path in standard format
    if not overwrite:
        full_path = (
            str(results_path)
            + "/"
            + str(cls_name)
            + "/Predictions/"
            + str(dataset)
            + "/testResample"
            + str(resample_id)
            + ".csv"
        )
        if os.path.exists(full_path):
            build_test = False
        if train_file:
            full_path = (
                str(results_path)
                + "/"
                + str(cls_name)
                + "/Predictions/"
                + str(dataset)
                + "/trainResample"
                + str(resample_id)
                + ".csv"
            )
            if os.path.exists(full_path):
                train_file = False
        if train_file is False and build_test is False:
            return

    # currently only works with .ts
    trainX, trainY = load_ts(problem_path + dataset + "/" + dataset + "_TRAIN" + format)
    testX, testY = load_ts(problem_path + dataset + "/" + dataset + "_TEST" + format)
    if resample_id != 0:
        trainX, trainY, testX, testY = stratified_resample(
            trainX, trainY, testX, testY, resample_id
        )
    run_clustering_experiment(
        trainX,
        clusterer,
        trainY=trainY,
        testX=testX,
        testY=testY,
        cls_name=cls_name,
        dataset_name=dataset,
        results_path=results_path,
    )


def _results_present_full_path(path, dataset, res):
    """Duplicate: check if results are present already without an estimator input."""
    full_path = f"{path}/Predictions/{dataset}/testResample{res}.csv"
    full_path2 = f"{path}/Predictions/{dataset}/trainResample{res}.csv"
    if os.path.exists(full_path) and os.path.exists(full_path2):
        return True
    return False


def config_clusterer(clusterer: str, **kwargs):
    """Config clusterer."""
    if clusterer == "kmeans":
        cls = TimeSeriesKMeans(**kwargs)
    elif clusterer == "kmedoids":
        cls = TimeSeriesKMedoids(**kwargs)
    return cls


def tune_window(metric: str, train_X, n_clusters):
    """Tune window."""
    best_w = 0
    best_score = sys.float_info.max
    for w in np.arange(0.0, 0.2, 0.01):
        cls = TimeSeriesKMeans(
            metric=metric, distance_params={"window": w}, n_clusters=n_clusters
        )
        cls.fit(train_X)
        preds = cls.predict(train_X)
        clusters = len(np.unique(preds))
        if clusters <= 1:
            score = sys.float_info.max
        else:
            score = davies_bouldin_score(train_X, preds)
        print(f" Number of clusters ={clusters} window = {w} score  = {score}")  # noqa
        if score < best_score:
            best_score = score
            best_w = w
    print("best window =", best_w, " with score ", best_score)  # noqa
    return best_w


def tune_msm(train_X, n_clusters):
    """Tune window for MSM."""
    best_c = 0
    best_score = sys.float_info.max
    for c in np.arange(0.0, 5.0, 0.25):
        cls = TimeSeriesKMeans(
            metric="msm", distance_params={"c": c}, n_clusters=n_clusters
        )
        cls.fit(train_X)
        preds = cls.predict(train_X)
        clusters = len(np.unique(preds))
        if clusters <= 1:
            score = sys.float_info.max
        else:
            score = davies_bouldin_score(train_X, preds)
        print(f" Number of clusters ={clusters} c parameter = {c} score  = {score}")  #
        # noqa
        if score < best_score:
            best_score = score
            best_c = c
    print("best c =", best_c, " with score ", best_score)  # noqa
    return best_c


def tune_wdtw(train_X, n_clusters):
    """Tune window for MSM."""
    best_g = 0
    best_score = sys.float_info.max
    for g in np.arange(0.0, 1.0, 0.05):
        cls = TimeSeriesKMeans(
            metric="wdtw", distance_params={"g": g}, n_clusters=n_clusters
        )
        cls.fit(train_X)
        preds = cls.predict(train_X)
        clusters = len(np.unique(preds))
        if clusters <= 1:
            score = sys.float_info.max
        else:
            score = davies_bouldin_score(train_X, preds)
        print(f" Number of clusters ={clusters} g parameter = {g} score  = {score}")  #
        # noqa
        if score < best_score:
            best_score = score
            best_g = g
    print("best g =", best_g, " with score ", best_score)  # noqa
    return best_g


def tune_twe(train_X, n_clusters):
    """Tune window for MSM."""
    best_nu = 0
    best_lambda = 0
    best_score = sys.float_info.max
    for nu in np.arange(0.0, 1.0, 0.25):
        for lam in np.arange(0.0, 1.0, 0.2):
            cls = TimeSeriesKMeans(
                metric="twe",
                distance_params={"nu": nu, "lmbda": lam},
                n_clusters=n_clusters,
            )
            cls.fit(train_X)
            preds = cls.predict(train_X)
            clusters = len(np.unique(preds))
            if clusters <= 1:
                score = sys.float_info.max
            else:
                score = davies_bouldin_score(train_X, preds)
            print(
                f" Number of clusters ={clusters} nu param = {nu} lambda para "
                f"= {lam} score  = {score}"
            )  #
            # noqa
            if score < best_score:
                best_score = score
                best_nu = nu
                best_lambda = lam
    print("best nu =", best_nu, f" lambda = {best_lambda} score ", best_score)  # noqa
    return best_nu, best_lambda


def tune_erp(train_X, n_clusters):
    """Tune window for MSM."""
    best_g = 0
    best_score = sys.float_info.max
    for g in np.arange(0.0, 2.0, 0.2):
        cls = TimeSeriesKMeans(
            metric="erp", distance_params={"g": g}, n_clusters=n_clusters
        )
        cls.fit(train_X)
        preds = cls.predict(train_X)
        clusters = len(np.unique(preds))
        if clusters <= 1:
            score = sys.float_info.max
        else:
            score = davies_bouldin_score(train_X, preds)
        print(f" Number of clusters ={clusters} g parameter = {g} score  = {score}")  #
        # noqa
        if score < best_score:
            best_score = score
            best_g = g
    print("best g =", best_g, " with score ", best_score)  # noqa
    return best_g


def tune_edr(train_X, n_clusters):
    """Tune window for MSM."""
    best_e = 0
    best_score = sys.float_info.max
    for e in np.arange(0.0, 0.2, 0.01):
        cls = TimeSeriesKMeans(
            metric="edr", distance_params={"epsilon": e}, n_clusters=n_clusters
        )
        cls.fit(train_X)
        preds = cls.predict(train_X)
        clusters = len(np.unique(preds))
        if clusters <= 1:
            score = sys.float_info.max
        else:
            score = davies_bouldin_score(train_X, preds)
        print(
            f" Number of clusters ={clusters} epsilon parameter = {e} score  ="
            f" {score}"
        )  #
        # noqa
        if score < best_score:
            best_score = score
            best_e = e
    print("best e =", best_e, " with score ", best_score)  # noqa
    return best_e


def tune_lcss(train_X, n_clusters):
    """Tune window for MSM."""
    best_e = 0
    best_score = sys.float_info.max
    for e in np.arange(0.0, 0.2, 0.01):
        cls = TimeSeriesKMeans(
            metric="lcss", distance_params={"epsilon": e}, n_clusters=n_clusters
        )
        cls.fit(train_X)
        preds = cls.predict(train_X)
        clusters = len(np.unique(preds))
        if clusters <= 1:
            score = sys.float_info.max
        else:
            score = davies_bouldin_score(train_X, preds)
        print(
            f" Number of clusters ={clusters} epsilon parameter = {e} score  ="
            f" {score}"
        )  #
        # noqa
        if score < best_score:
            best_score = score
            best_e = e
    print("best e =", best_e, " with score ", best_score)  # noqa
    return best_e


def _recreate_results(trainX, trainY):
    from sklearn.metrics import adjusted_rand_score

    clst = TimeSeriesKMeans(
        averaging_method="mean",
        metric="dtw",
        distance_params={"window": 0.2},
        n_clusters=len(set(train_Y)),
        random_state=1,
        verbose=True,
    )
    clst.fit(trainX)
    preds = clst.predict(trainY)
    score = adjusted_rand_score(trainY, preds)
    print("Score = ", score)  # noqa


def tune_cls(distance, train_X, n_clusters):
    """Tune clusterer."""
    best_init = "kmeans++"
    best_score = sys.float_info.max

    for init in ["kmeans++", "random", "forgy"]:
        cls = TimeSeriesKMedoids(
            init_algorithm=init,
            metric=distance,
            n_clusters=len(set(train_Y)),
        )
        cls.fit(train_X)
        preds = cls.predict(train_X)
        clusters = len(np.unique(preds))
        if clusters <= 1:
            score = sys.float_info.max
        else:
            score = davies_bouldin_score(train_X, preds)
        print(
            f" Number of clusters ={clusters} init alg = {init} score  =" f" {score}"
        )  #
        # noqa
        if score < best_score:
            best_score = score
            best_init = init
    return best_init

def get_distance_defaults(train_data: np.ndarray, dist_name: str) -> dict:
    if dist_name == "dtw" or dist_name == "ddtw":
        return {}
    if dist_name == "lcss":
        return {"epsilon": 1.}
    if dist_name == "erp":
        return {"g": train_data.std(axis=0).sum()}
    if dist_name == "msm":
        return {"c": 1., "independent": True}
    if dist_name == "edr":
        return {"epsilon": None}
    if dist_name == "twe":
        return {"nu": 0.001, "lmbda": 1.}


if __name__ == "__main__":
    """Example simple usage, with args input via script or hard coded for testing."""
    numba.set_num_threads(1)

    tune = False
    normalise = True
    if (
        sys.argv is not None and sys.argv.__len__() > 1
    ):  # cluster run, this is fragile, requires all args atm
        data_dir = sys.argv[1]
        results_dir = sys.argv[2]
        dataset = sys.argv[3]
        resample = int(sys.argv[4])
        distance = sys.argv[5]
        init = sys.argv[6]
        if len(sys.argv) > 7:
            normalise = sys.argv[7].lower() == "true"
        if len(sys.argv) > 8:
            tune = sys.argv[8].lower() == "true"
    else:  # Local run
        print(" Local Run")  # noqa
        data_dir = "/home/chris/Documents/Datasets/Univariate_ts/"
        dataset = "Chinatown"
        init = "random"
        results_dir = "/home/chris/Documents/Results/temp/"
        resample = 0
        distance = "msm"
        normalise = True
        tune = False

    if _results_present_full_path(results_dir + "/" + distance, dataset, resample):
        print(
            f"Ignoring dataset{dataset}, results already present at {results_dir}"
        )  # noqa
    else:
        print(  # noqa
            f" Running {dataset} resample {resample} normalised = {normalise} "  # noqa
            f"distance = {distance} "  # noqa
            f"tune window = {tune} results path = {results_dir}"
        )  # noqa

    train_X, train_Y = load_ts(f"{data_dir}/{dataset}/{dataset}_TRAIN.ts")
    test_X, test_Y = load_ts(f"{data_dir}/{dataset}/{dataset}_TEST.ts")
    test_X = test_X.squeeze()
    train_X = train_X.squeeze()

    distance_params = get_distance_defaults(train_X, distance)
    if normalise:
        from sklearn.preprocessing import StandardScaler

        s = StandardScaler()
        train_X = s.fit_transform(train_X.T)
        train_X = train_X.T
        test_X = s.fit_transform(test_X.T)
        test_X = test_X.T

    average_params = {
        "metric": distance,
        **distance_params.copy()
    }

    clst = TimeSeriesKMeans(
        n_clusters=len(set(train_Y)),
        metric=distance,
        distance_params=distance_params,
        init_algorithm=init,
        n_init=1,
        max_iter=300,
        averaging_method="dba",
        average_params=average_params,
        random_state=resample + 1,
    )
    print(f" Window parameters for {distance_params}")
    run_clustering_experiment(
        train_X,
        clst,
        results_path=results_dir,
        trainY=train_Y,
        testX=test_X,
        testY=test_Y,
        cls_name=distance,
        dataset_name=dataset,
        resample_id=resample,
        overwrite=False,
    )
    print("done")  # noqa