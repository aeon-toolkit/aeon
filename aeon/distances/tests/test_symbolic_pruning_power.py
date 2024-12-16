"""Test MinDist functions of symbolic representations."""

import itertools
import os
from warnings import simplefilter

import matplotlib
import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads
from scipy.stats import zscore

from aeon.distances._dft_sfa_mindist import dft_sfa_mindist
from aeon.distances._paa_sax_mindist import paa_sax_mindist
from aeon.transformations.collection.dictionary_based import SAX, SFAFast

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=UserWarning)

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
    # "Chinatown"
    "ArrowHead",
    "Beef",
    # "BeetleFly",
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


DATA_PATH = "/Users/bzcschae/workspace/UCRArchive_2018/"
server = False

if os.path.exists(DATA_PATH):
    DATA_PATH = "/Users/bzcschae/workspace/UCRArchive_2018/"
    used_dataset = dataset_names
# server
else:
    DATA_PATH = "/vol/fob-wbib-vol2/wbi/schaefpa/sktime/datasets/UCRArchive_2018"
    server = True
    used_dataset = dataset_names_full


@njit(cache=True, fastmath=True, parallel=True)
def compute_distances(
    queries,
    samples,
    PAA_queries,
    SAX_samples,
    SAX_breakpoints,
    all_breakpoints,
    all_dfts,
    all_words,
    method_names,
):
    """Compute lower bounding distances."""
    pruning_power = np.zeros((queries.shape[0], len(method_names)), dtype=np.float64)

    for i in prange(queries.shape[0]):
        # ED first
        nn_dist = np.inf
        for j in range(samples.shape[0]):
            ed = np.linalg.norm(queries[i] - samples[j])
            if ed != 0:
                nn_dist = min(nn_dist, ed)

        # used for pruning
        squared_lower_bound = nn_dist**2

        for j in range(samples.shape[0]):
            # SAX-PAA Min-Distance
            min_dist = paa_sax_mindist(
                PAA_queries[i],
                SAX_samples[j],
                SAX_breakpoints,
                samples.shape[-1],
                squared_lower_bound=squared_lower_bound,
            )
            if min_dist > nn_dist:
                pruning_power[i][0] += 1

        for a in range(all_dfts.shape[0]):
            for j in range(samples.shape[0]):
                # DFT-SFA Min-Distance variants
                min_dist = dft_sfa_mindist(
                    all_dfts[a][i],
                    all_words[a][j],
                    all_breakpoints[a],
                    squared_lower_bound=squared_lower_bound,
                )

                if min_dist > nn_dist:
                    pruning_power[i][a + 1] += 1

        # for i, mind in enumerate(mindist):
        #    if mind > ed:
        #        #print(f"mindist {method_names[i]} is:\t {mind} but ED is: \t {ed}")

    pruning_power = np.sum(pruning_power, axis=0)
    for i in range(len(pruning_power)):
        pruning_power[i] /= samples.shape[0] * queries.shape[0]

    return pruning_power


all_threads = 128
n_segments = 16
alphabet_sizes = [256, 128, 64, 32, 16, 8, 4, 2]
all_csv_scores = {}
set_num_threads(all_threads)

for alphabet_size in alphabet_sizes:
    all_csv_scores[alphabet_size] = []

for alphabet_size in alphabet_sizes:
    for dataset_name in used_dataset:
        csv_scores = all_csv_scores[alphabet_size]

        # print("Running: ", dataset_name, alphabet_size)

        X_train, y_train = load_from_ucr_tsv_to_dataframe_plain(
            os.path.join(DATA_PATH, dataset_name, dataset_name + "_TRAIN.tsv")
        )
        X_test, y_test = load_from_ucr_tsv_to_dataframe_plain(
            os.path.join(DATA_PATH, dataset_name, dataset_name + "_TEST.tsv")
        )

        # print("\tDataset\t", dataset_name)
        # print("\tData Shape\t", X_train.shape)
        # print("\tQuery Shape\t", X_test.shape)

        # ignore all future warnings
        simplefilter(action="ignore", category=FutureWarning)
        simplefilter(action="ignore", category=UserWarning)

        X_train.fillna(0, inplace=True)
        X_test.fillna(0, inplace=True)
        X_train = zscore(X_train.squeeze(), axis=1).values
        X_test = zscore(X_test.squeeze(), axis=1).values

        SAX_transform = SAX(n_segments=n_segments, alphabet_size=alphabet_size)
        SAX_train = SAX_transform.fit_transform(X_train).squeeze()
        PAA_test = SAX_transform._get_paa(X_test).squeeze()
        # print("\tSAX done.")

        histograms = ["equi-width", "equi-depth"]
        variances = [True, False]

        method_names = ["isax"]
        all_breakpoints = []
        all_dfts = []
        all_words = []

        for histogram, variance in itertools.product(histograms, variances):
            sfa = SFAFast(
                word_length=n_segments,
                alphabet_size=alphabet_size,
                window_size=X_train.shape[-1],
                binning_method=histogram,
                norm=True,
                variance=variance,
                lower_bounding_distances=True,
                n_jobs=all_threads,
            )

            sfa.fit(X_train)
            X_dfts = sfa.transform_mft(X_test).squeeze()
            Y_words = sfa.transform_words(X_train).squeeze()
            all_breakpoints.append(sfa.breakpoints.astype(np.float64))
            all_dfts.append(X_dfts.astype(np.float64))
            all_words.append(Y_words.astype(np.int32))

            method_names.append(f"sfa_{histogram}_{variance}")
            # print(f"\tSFA {histogram} {variance} done.")

        sum_scores = {}
        for method_name in method_names:
            sum_scores[method_name] = {
                "dataset": [],
                "pruning_power": [],
            }

        pruning_power = compute_distances(
            X_test,
            X_train,
            PAA_test,
            SAX_train,
            SAX_transform.breakpoints,
            np.array(all_breakpoints),
            np.array(all_dfts),
            np.array(all_words),
            np.array(method_names),
        )

        for a, method_name in enumerate(method_names):
            sum_scores[method_name]["dataset"].append(dataset_name)
            sum_scores[method_name]["pruning_power"].append(pruning_power[a])
            csv_scores.append((method_name, dataset_name, pruning_power[a]))

        # print(f"\n\n---- Results using {alphabet_size}-----")
        # for name, _ in sum_scores.items():
        #    print(
        #        f"---- Name {name},\tPrP: {
        #          np.round(sum_scores[name]['pruning_power'], 5)}"
        #    )

        # if server:
        pd.DataFrame.from_records(
            csv_scores,
            columns=[
                "Method",
                "Dataset",
                "Pruning_Power",
            ],
        ).to_csv(
            f"logs/pp_all_ucr_{n_segments}_{alphabet_size}" f"-18-11-24.csv", index=None
        )
