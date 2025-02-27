"""Test MinDist functions of symbolic representations."""

import itertools
import os
from warnings import simplefilter

import matplotlib
import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads
from scipy.stats import zscore

from aeon.distances.mindist._dft_sfa import mindist_dft_sfa_distance
from aeon.distances.mindist._paa_sax import mindist_paa_sax_distance
from aeon.transformations.collection.dictionary_based import SAX, SFAWhole

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
    # "Chinatown",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "Car",
    "CBF",
    "Coffee",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "ECG200",
    "ECGFiveDays",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "GunPoint",
    "ItalyPowerDemand",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    "OliveOil",
    "Plane",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "SyntheticControl",
    "TwoLeadECG",
    "Wine",
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


# configuration
all_threads = 128
n_segments = 16
alphabet_sizes = [256, 128, 64, 32, 16, 8, 4, 2]

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


@njit(cache=True, fastmath=True, parallel=True)
def compute_distances(
    queries,
    samples,
    PAA_queries,
    SAX_samples,
    SAX_breakpoints,
    # sfa_transforms,
    all_breakpoints,
    all_dfts,
    all_words,
    method_names,
):
    """Compute lower bounding distances."""
    p = 2

    tightness = np.zeros((queries.shape[0], len(method_names)), dtype=np.float64)
    for i in prange(queries.shape[0]):

        eds = np.zeros((samples.shape[0]), dtype=np.float32)
        for j in range(samples.shape[0]):
            eds[j] = np.linalg.norm(queries[i] - samples[j], ord=p)

        # SAX-PAA Min-Distance
        for j in range(samples.shape[0]):
            md = mindist_paa_sax_distance(
                PAA_queries[i],
                SAX_samples[j],
                SAX_breakpoints,
                samples.shape[-1],
                p=p,
            )
            if eds[j] > 0:
                tightness[i][0] += md / eds[j] / samples.shape[0]

            # if md > eds[j]:
            #     print(
            #         f"mindist {method_names[a]} is:\t {md} "
            #         f"but ED is:\t {eds[j]} \t Pos: {i}, {j}"
            #     )

        # DFT-SFA Min-Distance variants
        for a in range(all_dfts.shape[0]):
            for j in range(samples.shape[0]):
                md = mindist_dft_sfa_distance(
                    all_dfts[a][i],
                    all_words[a][j],
                    all_breakpoints[a],
                )
                if eds[j] > 0:
                    tightness[i][a + 1] += md / eds[j] / samples.shape[0]

                # if md > eds[j]:
                #     print(f"mindist {method_names[a+1]} is:\t "+str(md)+" "
                #          f"but ED is:\t "+str(eds[j])+f" \t Pos: {i}, {j}")
                #     assert False

    tightness = np.sum(tightness, axis=0)
    for i in range(len(tightness)):
        tightness[i] /= queries.shape[0]

    return tightness


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

        histograms = ["equi-width"]  # , "equi-depth"
        variances = [True]  # , False
        dyn_alphabets = [True]  # , False

        method_names = ["isax"]
        all_breakpoints = []
        all_dfts = []
        all_words = []

        for histogram, variance, dyn_alphabet in itertools.product(
            histograms, variances, dyn_alphabets
        ):
            # sfa = SFAFast(
            #     word_length=n_segments,
            #     alphabet_size=alphabet_size,
            #     window_size=X_train.shape[-1],
            #     binning_method=histogram,
            #     norm=True,
            #     variance=variance,
            #     lower_bounding_distances=True,
            #     learn_alphabet_sizes=dyn_alphabet,
            #     n_jobs=all_threads,
            # )

            sfa = SFAWhole(
                word_length=n_segments,
                alphabet_size=alphabet_size,
                binning_method=histogram,
                variance=variance,
                norm=True,
                learn_alphabet_sizes=dyn_alphabet,
                n_jobs=all_threads,
            )

            sfa.fit(X_train)
            # X_dfts = sfa.transform_mft(X_test).squeeze()
            # Y_words = sfa.transform_words(X_train).squeeze()

            _, X_dfts = sfa.transform_words(X_test)
            Y_words, _ = sfa.transform_words(X_train)
            all_breakpoints.append(sfa.breakpoints.astype(np.float64))
            all_dfts.append(X_dfts.astype(np.float64))
            all_words.append(Y_words.astype(np.int32))

            method_names.append(f"sfa_{histogram}_{variance}_{dyn_alphabet}")
            # print(f"\tSFA {histogram} {variance} done.")

        sum_scores = {}
        for method_name in method_names:
            sum_scores[method_name] = {
                "dataset": [],
                "tightness": [],
            }

        # print("\tTransformation done. Computing Distances")

        tightness = compute_distances(
            X_test,
            X_train,
            PAA_test,
            SAX_train,
            SAX_transform.breakpoints,
            all_breakpoints,
            np.array(all_dfts),
            np.array(all_words),
            np.array(method_names),
        )

        for i, method_name in enumerate(method_names):
            sum_scores[method_name]["dataset"].append(dataset_name)
            sum_scores[method_name]["tightness"].append(tightness[i])
            csv_scores.append((method_name, dataset_name, tightness[i]))

        # print(f"\n\n---- Results using {alphabet_size}-----")
        # for name, _ in sum_scores.items():
        #    print(
        #        f"---- Name {name}, \t tlb:
        #        {np.round(sum_scores[name]['tightness'], 3)}"
        #    )

        # if server:
        pd.DataFrame.from_records(
            csv_scores,
            columns=[
                "Method",
                "Dataset",
                "TLB",
            ],
        ).to_csv(
            f"logs/tlb_all_ucr_{n_segments}_{alphabet_size}-26_02_25.csv", index=None
        )
