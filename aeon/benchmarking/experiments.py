"""Functions to perform classification and clustering experiments.

Results are saved a standardised format used by both tsml and aeon.
"""

__maintainer__ = []
__all__ = [
    "run_clustering_experiment",
    "load_and_run_clustering_experiment",
    "run_classification_experiment",
    "load_and_run_classification_experiment",
]


import os
import time
from datetime import datetime
from itertools import chain

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_random_state

from aeon.datasets import write_results_to_uea_format
from aeon.datasets._data_loaders import load_from_tsfile as load_ts


def stratified_resample(X_train, y_train, X_test, y_test, random_state=None):
    """Stratified resample data without replacement using a random state.

    Reproducable resampling. Combines train and test, resamples to get the same class
    distribution, then returns new train and test.

    Parameters
    ----------
    X_train : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints), list of
    shape[n_cases] of 2D arrays shape (n_channels,n_timepoints_i), or pd.DataFrame
        train data attributes.
    y_train : np.array
        train data class labels.
    X_test : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints), list of
    shape[n_cases] of 2D arrays shape (n_channels,n_timepoints_i), or pd.DataFrame
        test data attributes.
    y_test : np.array
        test data class labels as np array.
    random_state : int
        seed to enable reproduceable resamples

    Returns
    -------
    new train and test attributes and class labels.
    """
    random_state = check_random_state(random_state)
    all_labels = np.concatenate([y_train, y_test], axis=0)
    if isinstance(X_train, pd.DataFrame):
        all_data = pd.concat([X_train, X_test], ignore_index=True)
    elif isinstance(X_train, list):
        all_data = list(x for x in chain(X_train, X_test))
    else:  # 3D or 2D numpy
        all_data = np.concatenate([X_train, X_test], axis=0)

    # count class occurrences
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test = np.unique(y_test)

    # haven't built functionality to deal with classes that exist in
    # test but not in train
    assert set(unique_train) == set(unique_test)

    new_train_indices = []
    new_test_indices = []
    for label, count_train in zip(unique_train, counts_train):
        class_indexes = np.argwhere(all_labels == label).ravel()

        # randomizes the order and partition into train and test
        random_state.shuffle(class_indexes)
        new_train_indices.extend(class_indexes[:count_train])
        new_test_indices.extend(class_indexes[count_train:])

    if isinstance(X_train, pd.DataFrame):
        new_X_train = all_data.iloc[new_train_indices]
        new_X_test = all_data.iloc[new_test_indices]
        new_X_train = new_X_train.reset_index(drop=True)
        new_X_test = new_X_test.reset_index(drop=True)
    elif isinstance(X_train, list):
        new_X_train = list(all_data[i] for i in new_train_indices)
        new_X_test = list(all_data[i] for i in new_test_indices)
    else:  # 3D or 2D numpy
        new_X_train = all_data[new_train_indices]
        new_X_test = all_data[new_test_indices]

    new_y_train = all_labels[new_train_indices]
    new_y_test = all_labels[new_test_indices]

    return new_X_train, new_y_train, new_X_test, new_y_test


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
    this method will always overwrite existing results, check before calling or use
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
    start = int(round(time.time() * 1000))
    train_preds = clusterer.predict(trainX)
    build_time = int(round(time.time() * 1000)) - start
    train_probs = clusterer.predict_proba(trainX)

    start = int(round(time.time() * 1000))
    test_preds = clusterer.predict(testX)
    test_time = int(round(time.time() * 1000)) - start
    test_probs = clusterer.predict_proba(testX)
    second = str(clusterer.get_params())
    second.replace("\n", " ")
    second.replace("\r", " ")
    length = np.unique(trainY).size
    third = f"0,{build_time},{test_time},-1,-1,{length},{length}"
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

    #        preds = form_cluster_list(clusters, len(testY))
    if "Composite" in cls_name:
        second = "Para info too long!"
    else:
        second = str(clusterer.get_params())
        second.replace("\n", " ")
        second.replace("\r", " ")
    len = np.unique(trainY)
    third = f"0,{build_time},{test_time},-1,-1,{len},{len}"
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
        format, see   examples/datasets/data_loading.ipynb
    train_file: boolean, default = False
        whether to generate train files or not. If true, it performs a 10xCV on the
        train and saves
    """
    if cls_name is None:
        cls_name = type(clusterer).__name__

    # Set up the file path in standard format
    if not overwrite:
        full_path = (
            f"{results_path}/{cls_name}/Predictions/{dataset}/testResample"
            f"{resample_id}.csv"
        )
        if os.path.exists(full_path):
            build_test = False
        if train_file:
            full_path = (
                f"{results_path}/{cls_name}/Predictions/"
                f"{dataset}/trainResample{resample_id}.csv"
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


def run_classification_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    classifier,
    results_path,
    cls_name="",
    dataset="",
    resample_id=0,
    train_file=False,
    test_file=True,
):
    """Run a classification experiment and save the results to file.

    Method to run a basic experiment and write the results to files called
    testFold<resampleID>.csv and, if required, trainFold<resampleID>.csv.

    Parameters
    ----------
    X_train : pd.DataFrame or np.array
        The data to train the classifier.
    y_train : np.array, default = None
        Training data class labels.
    X_test : pd.DataFrame or np.array, default = None
        The data used to test the trained classifier.
    y_test : np.array, default = None
        Testing data class labels.
    classifier : BaseClassifier
        Classifier to be used in the experiment.
    results_path : str
        Location of where to write results. Any required directories will be created.
    cls_name : str, default=""
        Name of the classifier.
    dataset : str, default=""
        Name of problem.
    resample_id : int, default=0
        Seed for resampling. If set to 0, the default train/test split from file is
        used. Also used in output file name.
    train_file : bool, default=False
        Whether to generate train files or not. If true, it performs a 10-fold
        cross-validation on the train data and saves. If the classifier can produce its
        own estimates, those are used instead.
    test_file : bool, default=True:
         Whether to generate test files or not. If the classifier can generate its own
         train probabilities, the classifier will be built but no file will be output.
    """
    if not test_file and not train_file:
        raise Exception(
            "Both test_file and train_file are set to False. "
            "At least one must be output."
        )

    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    encoder_dict = {label: i for i, label in enumerate(le.classes_)}
    classifier_train_probs = train_file and callable(
        getattr(classifier, "_get_train_probs", None)
    )
    build_time = -1

    if test_file or classifier_train_probs:
        start = int(round(time.time() * 1000))
        classifier.fit(X_train, y_train)
        build_time = int(round(time.time() * 1000)) - start

    if test_file:
        start = int(round(time.time() * 1000))
        probs = classifier.predict_proba(X_test)
        test_time = int(round(time.time() * 1000)) - start

        if "composite" in cls_name.lower():
            second = "Para info too long!"
        else:
            second = str(classifier.get_params())
        second.replace("\n", " ")
        second.replace("\r", " ")

        # Line 3 format:
        preds = classifier.classes_[np.argmax(probs, axis=1)]
        acc = accuracy_score(y_test, preds)
        third = (
            str(acc)  # 1. accuracy
            + ","
            + str(build_time)  # 2. fit time
            + ","
            + str(test_time)  # 3. predict time
            + ",-1,-1,"  # 4. 5. benchmark time, memory (to do)
            + str(len(classifier.classes_))  # 6. number of classes
            + ",,-1,-1"  # 7. 8. 9.
        )

        write_results_to_uea_format(
            second_line=second,
            third_line=third,
            first_line_comment="PREDICTIONS,Generated by experiments.py on "
            + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            + ". Encoder dictionary: "
            + str(encoder_dict),
            timing_type="MILLISECONDS",
            output_path=results_path,
            estimator_name=cls_name,
            resample_seed=resample_id,
            y_pred=preds,
            predicted_probs=probs,
            dataset_name=dataset,
            y_true=y_test,
            split="TEST",
            full_path=False,
        )

    if train_file:
        start = int(round(time.time() * 1000))
        if classifier_train_probs:  # Normally can only do this if test has been built
            train_probs = classifier._get_train_probs(X_train, y_train)
        else:
            cv_size = 10
            _, counts = np.unique(y_train, return_counts=True)
            min_class = np.min(counts)
            if min_class < cv_size:
                cv_size = min_class

            train_probs = cross_val_predict(
                classifier, X_train, y=y_train, cv=cv_size, method="predict_proba"
            )
        train_time = int(round(time.time() * 1000)) - start

        if "composite" in cls_name.lower():
            second = "Para info too long!"
        else:
            second = str(classifier.get_params())
        second.replace("\n", " ")
        second.replace("\r", " ")

        train_preds = classifier.classes_[np.argmax(train_probs, axis=1)]
        train_acc = accuracy_score(y_train, train_preds)
        third = (
            str(train_acc)
            + ","
            + str(build_time)
            + ",-1,-1,-1,"
            + str(len(classifier.classes_))
            + ",,"
            + str(train_time)
            + ","
            + str(build_time + train_time)
        )

        write_results_to_uea_format(
            second_line=second,
            third_line=third,
            first_line_comment="PREDICTIONS,Generated by classification_experiments.py "
            "on "
            + datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
            + ". Encoder dictionary: "
            + str(encoder_dict),
            timing_type="MILLISECONDS",
            output_path=results_path,
            estimator_name=cls_name,
            resample_seed=resample_id,
            y_pred=train_preds,
            predicted_probs=train_probs,
            dataset_name=dataset,
            y_true=y_train,
            split="TRAIN",
            full_path=False,
        )


def load_and_run_classification_experiment(
    problem_path,
    results_path,
    dataset,
    classifier,
    resample_id=0,
    cls_name=None,
    overwrite=False,
    build_train=False,
    predefined_resample=False,
):
    """Load a dataset and run a classification experiment.

    Method to run a basic experiment and write the results to files called
    testFold<resampleID>.csv and, if required, trainFold<resampleID>.csv.

    Parameters
    ----------
    problem_path : str
        Location of problem files, full path.
    results_path : str
        Location of where to write results. Any required directories will be created.
    dataset : str
        Name of problem. Files must be  <problem_path>/<dataset>/<dataset>+"_TRAIN.ts",
        same for "_TEST".
    classifier : BaseClassifier
        Classifier to be used in the experiment, if none is provided one is selected
        using cls_name using resample_id as a seed.
    cls_name : str, default = None
        Name of classifier used in writing results. If none the name is taken from
        the classifier
    resample_id : int, default=0
        Seed for resampling. If set to 0, the default train/test split from file is
        used. Also used in output file name.
    overwrite : bool, default=False
        If set to False, this will only build results if there is not a result file
        already present. If True, it will overwrite anything already there.
    build_train : bool, default=False
        Whether to generate train files or not. If true, it performs a 10-fold
        cross-validation on the train data and saves. If the classifier can produce its
        own estimates, those are used instead.
    predefined_resample : bool, default=False
        Read a predefined resample from file instead of performing a resample. If True
        the file format must include the resample_id at the end of the dataset name i.e.
        <problem_path>/<dataset>/<dataset>+<resample_id>+"_TRAIN.ts".
    """
    if cls_name is None:
        cls_name = type(classifier).__name__
    # Check which files exist, if both exist, exit
    build_test = True
    if not overwrite:
        full_path = (
            results_path
            + "/"
            + cls_name
            + "/Predictions/"
            + dataset
            + "/testResample"
            + str(resample_id)
            + ".csv"
        )

        if os.path.exists(full_path):
            build_test = False

        if build_train:
            full_path = (
                results_path
                + "/"
                + cls_name
                + "/Predictions/"
                + dataset
                + "/trainResample"
                + str(resample_id)
                + ".csv"
            )

            if os.path.exists(full_path):
                build_train = False

        if build_train is False and not build_test:
            return

    if predefined_resample:
        X_train, y_train = load_ts(
            problem_path + dataset + "/" + dataset + str(resample_id) + "_TRAIN.ts"
        )
        X_test, y_test = load_ts(
            problem_path + dataset + "/" + dataset + str(resample_id) + "_TEST.ts"
        )
    else:
        X_train, y_train = load_ts(problem_path + dataset + "/" + dataset + "_TRAIN.ts")
        X_test, y_test = load_ts(problem_path + dataset + "/" + dataset + "_TEST.ts")
        if resample_id != 0:
            X_train, y_train, X_test, y_test = stratified_resample(
                X_train, y_train, X_test, y_test, resample_id
            )

    run_classification_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        classifier,
        results_path,
        cls_name=cls_name,
        dataset=dataset,
        resample_id=resample_id,
        train_file=build_train,
        test_file=build_test,
    )
