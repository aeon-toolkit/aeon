"""Test EVAR of symbolic representations."""

import os
from warnings import simplefilter
from joblib import Parallel, delayed

import scipy
import numpy as np
import pandas as pd
from numba import set_num_threads, njit, objmode
from scipy.stats import zscore
from sklearn.decomposition import PCA

simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=UserWarning)


def explain_variance(X, model, n_components):
    result = np.zeros(n_components)
    for ii in range(n_components):
        X_trans = model.transform(X)
        X_trans_ii = np.zeros_like(X_trans)
        X_trans_ii[:, ii] = X_trans[:, ii]
        X_approx_ii = model.inverse_transform(X_trans_ii)
        result[ii] = 1 - (np.linalg.norm(X_approx_ii - X) /
                          np.linalg.norm(X - model.mean_)) ** 2
    return result


# @njit(fastmath=True, cache=True)
def _fast_fourier_transform(X, norm, dft_length, inverse_sqrt_win_size, norm_std=True):
    """Perform a discrete fourier transform using the fast fourier transform.

    if self.norm is True, then the first term of the DFT is ignored

    Input
    -------
    X : The training input samples.  array-like or sparse matrix of
    shape = [n_samps, num_atts]

    Returns
    -------
    1D array of fourier term, real_0,imag_0, real_1, imag_1 etc, length
    num_atts or
    num_atts-2 if if self.norm is True
    """
    # first two are real and imaginary parts
    start = 2 if norm else 0
    length = start + dft_length
    dft = np.zeros((len(X), length))  # , dtype=np.float64

    with objmode(X_ffts="complex128[:,:]"):
        X_ffts = scipy.fft.rfft(X, axis=1, workers=-1).astype(np.complex128)

    reals = np.real(X_ffts)  # float64[]
    imags = np.imag(X_ffts)  # float64[]
    count = (length // 2) * 2

    dft[:, 0:count:2] = reals[:, 0: length // 2]
    dft[:, 1:count:2] = imags[:, 0: length // 2]
    dft *= inverse_sqrt_win_size

    # apply z-normalization
    if norm_std:
        stds = np.zeros(len(X))
        for i in range(len(stds)):
            stds[i] = np.std(X[i])
        # stds = np.std(X, axis=1)  # not available in numba
        stds = np.where(stds < 1e-8, 1, stds)
        dft /= stds.reshape(-1, 1)

    return dft[:, start:]


dataset_names_full = [
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "BME",
    "Car",
    "CBF",
    "Chinatown",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "Crop",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "DodgerLoopDay",
    "DodgerLoopGame",
    "DodgerLoopWeekend",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "EthanolLevel",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "Fungi",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "GesturePebbleZ1",
    "GesturePebbleZ2",
    "GunPoint",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "HouseTwenty",
    "InlineSkate",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    # "MelbournePedestrian",            # iSAX mindist error?
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    "MixedShapesRegularTrain",
    "MixedShapesSmallTrain",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "PickupGestureWiimoteZ",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "PLAID",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    # "SmoothSubspace",         # SFA mindist error?
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UMD",
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
]

dataset_names = [
    "ItalyPowerDemand",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    # "BirdChicken",
    # "Car",
    # "CBF",
    # "Coffee",
    # "DiatomSizeReduction",
    # "DistalPhalanxOutlineAgeGroup",
    # "DistalPhalanxOutlineCorrect",
    # "DistalPhalanxTW",
    # "ECG200",
    # "ECGFiveDays",
    # "FaceAll",
    # "FaceFour",
    # "FacesUCR",
    # "GunPoint",
    # "ItalyPowerDemand",
    # "MiddlePhalanxOutlineAgeGroup",
    # "MiddlePhalanxOutlineCorrect",
    # "MiddlePhalanxTW",
    # "OliveOil",
    # "Plane",
    # "ProximalPhalanxOutlineAgeGroup",
    # "ProximalPhalanxOutlineCorrect",
    # "ProximalPhalanxTW",
    # "SonyAIBORobotSurface1",
    # "SonyAIBORobotSurface2",
    # "SyntheticControl",
    # "TwoLeadECG",
    # "Wine",
]


def load_from_ucr_tsv_to_dataframe_plain(full_file_path_and_name):
    """Load UCR datasets."""
    df = pd.read_csv(
        full_file_path_and_name,
        sep=r"\s+|\t+|\s+\t+|\t+\s+",
        engine="python",
        header=None,
    )
    y = df.pop(0).values
    df.columns -= 1
    return df, y

def process_n_components(
        n_components, X_train, X_test, dataset_name
):
    print("\tRunning: ", dataset_name, n_components)
    csv_scores = []

    sum_scores = {}
    method_names = ["PISA", "SPARTAN"]
    for method_name in method_names:
        sum_scores[method_name] = {
            "dataset": [],
            "EVAR_TRAIN": [],
            "EVAR_TEST": [],
        }

    # PISA: DFT + PCA
    n = X_train.shape[-1]
    dft_length = n // 2 + n % 2

    if n_components > dft_length:
        return n_components, sum_scores, csv_scores

    inverse_sqrt_win_size = 1.0 / np.sqrt(n)
    norm, norm_std = True, True

    X_train_dfts = _fast_fourier_transform(
        X_train, norm, dft_length, inverse_sqrt_win_size, norm_std=norm_std)
    X_test_dfts = _fast_fourier_transform(
        X_test, norm, dft_length, inverse_sqrt_win_size, norm_std=norm_std)

    pca_transform = PCA(
        n_components=n_components,
        svd_solver="auto"
    ).fit(X_train_dfts)
    X_dft = pca_transform.transform(X_train_dfts)
    explained_variance_train = pca_transform.explained_variance_ratio_.sum()
    explained_variance_test = explain_variance(X_test_dfts, pca_transform,
                                               X_dft.shape[-1]).sum()

    method_name = "PISA"
    sum_scores[method_name]["dataset"].append(dataset_name)
    sum_scores[method_name]["EVAR_TRAIN"].append(explained_variance_train)
    sum_scores[method_name]["EVAR_TEST"].append(explained_variance_test)
    csv_scores.append((method_name, dataset_name,
                       explained_variance_train,
                       explained_variance_test))

    # SPARTAN: PCA
    pca_transform = PCA(
        n_components=n_components,
        svd_solver="auto"
    ).fit(X_train)
    X_pca = pca_transform.transform(X_train)
    explained_variance_train = pca_transform.explained_variance_ratio_.sum()
    explained_variance_test = explain_variance(X_test, pca_transform,
                                               X_pca.shape[-1]).sum()

    method_name = "SPARTAN"
    sum_scores[method_name]["dataset"].append(dataset_name)
    sum_scores[method_name]["EVAR_TRAIN"].append(explained_variance_train)
    sum_scores[method_name]["EVAR_TEST"].append(explained_variance_test)
    csv_scores.append((method_name, dataset_name,
                       explained_variance_train,
                       explained_variance_test))

    return n_components, sum_scores, csv_scores



# configuration
all_threads = os.cpu_count() - 1
all_n_components = np.arange(2, 18, 2, dtype=np.int32)
all_csv_scores = {}
set_num_threads(all_threads)

DATA_PATH = "/Users/bzcschae/workspace/UCRArchive_2018/"
server = False

if os.path.exists(DATA_PATH):
    DATA_PATH = "/Users/bzcschae/workspace/UCRArchive_2018/"
    used_dataset = dataset_names
    alphabet_sizes = [16]
    all_threads = 8
# server
else:
    DATA_PATH = "/vol/fob-wbib-vol2/wbi/schaefpa/sktime/datasets/UCRArchive_2018"
    server = True
    used_dataset = dataset_names_full

for n_components in all_n_components:
    all_csv_scores[n_components] = []

for dataset_name in used_dataset:
    # print("\tDataset\t", dataset_name)
    # print("\tData Shape\t", X_train.shape)
    # print("\tQuery Shape\t", X_test.shape)

    X_train, y_train = load_from_ucr_tsv_to_dataframe_plain(
        os.path.join(DATA_PATH, dataset_name, dataset_name + "_TRAIN.tsv")
    )
    X_test, y_test = load_from_ucr_tsv_to_dataframe_plain(
        os.path.join(DATA_PATH, dataset_name, dataset_name + "_TEST.tsv")
    )

    # ignore all future warnings
    simplefilter(action="ignore", category=FutureWarning)
    simplefilter(action="ignore", category=UserWarning)

    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)
    X_train = zscore(X_train.squeeze(), axis=1).values
    X_test = zscore(X_test.squeeze(), axis=1).values

    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(process_n_components)(
            n_components, X_train, X_test, dataset_name
        )
        for n_components in all_n_components
    )

    for n_components, sum_scores, csv_score in results:
        if csv_score:  # Only process if not skipped
            print(f"\n\n---- Results using {dataset_name} {n_components} -----")
            for name, _ in sum_scores.items():
                print(f"---- Name {name}, EVAR:\t"
                      f"{sum_scores[name]['EVAR_TRAIN'][0]:0.3f}\t"
                      f"{sum_scores[name]['EVAR_TEST'][0]:0.3f}")
            all_csv_scores[n_components].extend(csv_score)

            pd.DataFrame.from_records(
                all_csv_scores[n_components],
                columns=[
                    "Method",
                    "Dataset",
                    "EVAR_TRAIN",
                    "EVAR_TEST",
                ],
            ).to_csv(
                f"logs/evar_all_ucr_{n_components}-04_07_25.csv", index=None
            )
