"""Test MinDist functions of symbolic representations."""

import itertools
import os
import time
from warnings import simplefilter

import numpy as np
import pandas as pd
from numba import njit, objmode, prange, set_num_threads
from scipy.stats import zscore

from aeon.distances.mindist._dft_sfa import mindist_dft_sfa_distance
from aeon.distances.mindist._paa_sax import mindist_paa_sax_distance
from aeon.distances.mindist._pca_spartan import mindist_pca_spartan_distance
from aeon.transformations.collection.dictionary_based import SAX, SPARTAN, SFAWhole

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
    sax_breakpoints,
    other_breakpoints,
    all_coeffs,
    all_words,
    method_names,
):
    """Compute lower bounding distances."""
    pruning_power = np.zeros((queries.shape[0], len(method_names)), dtype=np.float64)
    runtimes = np.zeros(len(method_names), dtype=np.float64)

    for i in prange(queries.shape[0]):
        # ED first
        nn_dist = np.inf
        eds = np.zeros((samples.shape[0]), dtype=np.float64)
        for j in range(samples.shape[0]):
            eds[j] = np.linalg.norm(queries[i] - samples[j])
        nn_dist = np.nanmin(eds)

        # used for pruning
        squared_lower_bound = nn_dist**2

        for a in prange(method_names.shape[0]):
            with objmode(start_time="f8"):
                start_time = time.time()

            for j in range(samples.shape[0]):
                if method_names[a].startswith("sax"):
                    # SAX-PAA Min-Distance
                    min_dist = mindist_paa_sax_distance(
                        all_coeffs[a][i],
                        all_words[a][j],
                        sax_breakpoints,
                        samples.shape[-1],
                        squared_lower_bound=squared_lower_bound,
                    )
                elif (method_names[a].startswith("sofa")) or (
                    method_names[a].startswith("sfa")
                ):
                    # DFT-SFA Min-Distance variants
                    min_dist = mindist_dft_sfa_distance(
                        all_coeffs[a][i],
                        all_words[a][j],
                        other_breakpoints[a - 1],
                        squared_lower_bound=squared_lower_bound,
                    )
                elif method_names[a].startswith("spartan"):
                    # SPARTAN Min-Distance
                    min_dist = mindist_pca_spartan_distance(
                        all_coeffs[a][i],
                        all_words[a][j],
                        other_breakpoints[a - 1],
                        squared_lower_bound=squared_lower_bound,
                    )
                else:
                    print(f"Unknown method name {method_names[a]}.")
                    continue

                if np.isnan(min_dist) or np.isinf(min_dist) or (min_dist > nn_dist):
                    pruning_power[i][a] += 1

                if ~np.isinf(min_dist) and (min_dist > eds[j]):
                    print(
                        f"mindist {method_names[a]} is:",
                        np.round(min_dist, 1),
                        f" but ED is: ",
                        np.round(eds[j], 1),
                        f" Pos: {i}, {j}",
                    )

            with objmode(end_time="f8"):
                end_time = time.time()

            runtimes[a] += end_time - start_time

    pruning_power = np.sum(pruning_power, axis=0)
    for i in range(len(pruning_power)):
        pruning_power[i] /= samples.shape[0] * queries.shape[0]

    return pruning_power, runtimes


all_threads = os.cpu_count() - 1
n_segments = 16
alphabet_sizes = [256, 128, 64, 32, 16, 8, 4, 2]
all_csv_scores = {}
set_num_threads(all_threads)

for alphabet_size in alphabet_sizes:
    all_csv_scores[alphabet_size] = []

for dataset_name in used_dataset:
    for alphabet_size in alphabet_sizes:
        csv_scores = all_csv_scores[alphabet_size]

        # print("Running: ", dataset_name, alphabet_size)

        X_train, y_train = load_from_ucr_tsv_to_dataframe_plain(
            os.path.join(DATA_PATH, dataset_name, dataset_name + "_TRAIN.tsv")
        )
        X_test, y_test = load_from_ucr_tsv_to_dataframe_plain(
            os.path.join(DATA_PATH, dataset_name, dataset_name + "_TEST.tsv")
        )

        print("\tDataset\t", dataset_name)
        # print("\tData Shape\t", X_train.shape)
        # print("\tQuery Shape\t", X_test.shape)

        # ignore all future warnings
        simplefilter(action="ignore", category=FutureWarning)
        simplefilter(action="ignore", category=UserWarning)

        X_train.fillna(0, inplace=True)
        X_test.fillna(0, inplace=True)
        X_train = zscore(X_train.squeeze(), axis=1).values
        X_test = zscore(X_test.squeeze(), axis=1).values

        histograms = ["equi-width"]
        allocation_methods = [
            "dynamic_programming",
            "linear_scale",
            "log_scale",
            "sqrt_scale",
        ]
        feature_selections = ["pca"]

        method_names = []
        other_breakpoints = []
        all_coeffs = []
        all_words = []

        sax = SAX(n_segments=n_segments, alphabet_size=alphabet_size)
        X_words = sax.fit_transform(X_train).squeeze()
        Y_paa = sax._get_paa(X_test).squeeze()
        sax_breakpoints = sax.breakpoints.astype(np.float64)
        all_coeffs.append(Y_paa.astype(np.float64))
        all_words.append(X_words.astype(np.int32))
        method_names.append("sax")

        sfa = SFAWhole(
            word_length=n_segments,
            alphabet_size=alphabet_size,
            binning_method="equi-width",
            feature_selection_strategy=None,
            alphabet_allocation_method=None,
            n_jobs=all_threads,
        ).fit(X_train)
        Y_words, _ = sfa.transform_words(X_train)
        _, X_dfts = sfa.transform_words(X_test)
        other_breakpoints.append(sfa.breakpoints.astype(np.float64))
        all_coeffs.append(X_dfts.astype(np.float64))
        all_words.append(Y_words.astype(np.int32))
        method_names.append("sfa")

        sfa = SFAWhole(
            word_length=n_segments,
            alphabet_size=alphabet_size,
            binning_method="equi-width",
            feature_selection_strategy="variance",
            alphabet_allocation_method=None,
            n_jobs=all_threads,
        ).fit(X_train)
        Y_words, _ = sfa.transform_words(X_train)
        _, X_dfts = sfa.transform_words(X_test)
        other_breakpoints.append(sfa.breakpoints.astype(np.float64))
        all_coeffs.append(X_dfts.astype(np.float64))
        all_words.append(Y_words.astype(np.int32))
        method_names.append("sofa")

        SPARTAN_transform = SPARTAN(
            word_length=n_segments,
            alphabet_size=alphabet_size,
            build_histogram=False,
            return_sparse=False,
        ).fit(X_train)
        train_words, _ = SPARTAN_transform.transform_words(X_train)
        _, test_pca = SPARTAN_transform.transform_words(X_test)
        other_breakpoints.append(SPARTAN_transform.breakpoints.astype(np.float64))
        all_coeffs.append(Y_pca.astype(np.float64))
        all_words.append(X_words.astype(np.int32))
        method_names.append("spartan")

        for histogram, fs_strategy, alloc_method in itertools.product(
            histograms, feature_selections, allocation_methods
        ):
            sfa = SFAWhole(
                word_length=n_segments,
                alphabet_size=alphabet_size,
                binning_method=histogram,
                feature_selection_strategy=fs_strategy,
                alphabet_allocation_method=alloc_method,
                n_jobs=all_threads,
            )
            sfa.fit(X_train)

            Y_words, _ = sfa.transform_words(X_train)
            _, X_dfts = sfa.transform_words(X_test)
            other_breakpoints.append(sfa.breakpoints.astype(np.float64))
            all_coeffs.append(X_dfts.astype(np.float64))
            all_words.append(Y_words.astype(np.int32))

            method_names.append(f"sofa_{histogram}_{fs_strategy}_{alloc_method}")

        sum_scores = {}
        for method_name in method_names:
            sum_scores[method_name] = {
                "dataset": [],
                "pruning_power": [],
                "runtime": [],
            }

        print("\tTransformation done. Computing Distances")

        pruning_power, runtimes = compute_distances(
            X_test,
            X_train,
            sax_breakpoints,
            other_breakpoints,
            np.array(all_coeffs),
            np.array(all_words),
            np.array(method_names),
        )

        for a, method_name in enumerate(method_names):
            sum_scores[method_name]["dataset"].append(dataset_name)
            sum_scores[method_name]["pruning_power"].append(pruning_power[a])
            sum_scores[method_name]["runtime"].append(runtimes[a])
            csv_scores.append(
                (method_name, dataset_name, pruning_power[a], runtimes[a])
            )

        print(f"\n\n---- Results using {alphabet_size}-----")
        for name, _ in sum_scores.items():
            print(
                f"---- Name {name},\tPrP: "
                f"{sum_scores[name]['pruning_power'][0]:0.3f}, "
                f"{sum_scores[name]['runtime'][0]:0.3f}"
            )

        # if server:
        pd.DataFrame.from_records(
            csv_scores,
            columns=[
                "Method",
                "Dataset",
                "Pruning_Power",
                "Runtime",
            ],
        ).to_csv(
            f"logs/pp_all_ucr_{n_segments}_{alphabet_size}-01-07-25-2.csv", index=None
        )
