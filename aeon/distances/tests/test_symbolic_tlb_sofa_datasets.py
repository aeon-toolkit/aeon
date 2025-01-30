"""Test MinDist functions of symbolic representations."""

import itertools
from warnings import simplefilter

import matplotlib
import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads

from aeon.distances.mindist._dft_sfa import mindist_dft_sfa_distance
from aeon.distances.mindist._paa_sax import mindist_paa_sax_distance
from aeon.transformations.collection.dictionary_based import SAX, SFAWhole

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=UserWarning)


def read(fp, dim, data_type=np.float32, count=100):
    """Read datasets from file."""
    if data_type != np.float32:
        a = np.fromfile(fp, dtype=data_type, count=dim * count)
        return a.reshape(-1, dim).copy().astype(np.float64, copy=False)
    else:
        return (
            np.fromfile(fp, dtype=np.float32, count=dim * count)
            .reshape(-1, dim)
            .astype(np.float32, copy=False)
        )


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
    tighness = np.zeros((queries.shape[0], len(method_names)), dtype=np.float64)
    for i in prange(queries.shape[0]):

        eds = np.zeros((samples.shape[0]), dtype=np.float32)
        for j in range(samples.shape[0]):
            eds[j] = np.linalg.norm(queries[i] - samples[j])

        # SAX-PAA Min-Distance
        for j in range(samples.shape[0]):
            md = mindist_paa_sax_distance(
                PAA_queries[i],
                SAX_samples[j],
                SAX_breakpoints,
                samples.shape[-1],
            )
            if eds[j] > 0:
                tighness[i][0] += md / eds[j] / samples.shape[0]

        # DFT-SFA Min-Distance variants
        for a in range(all_dfts.shape[0]):
            for j in range(samples.shape[0]):
                md = mindist_dft_sfa_distance(
                    all_dfts[a][i],
                    all_words[a][j],
                    all_breakpoints[a],
                )
                if eds[j] > 0:
                    tighness[i][a + 1] += md / eds[j] / samples.shape[0]

            # for a, mind in enumerate(mindist):
            #    if mind > ed:
            #        #print(f"mindist {method_names[a]} is:\t {mind}
            #        but ED is: \t {ed} \t Pos: {i}, {j}")
            #        # #print(f"Query std/mean: \t {np.std(queries[i])},
            #        {np.mean(queries[i])}")
            #        # #print(f"Sample std/mean:\t {np.std(samples[j])},
            #        {np.mean(samples[j])}")

    tighness = np.sum(tighness, axis=0)
    for i in range(len(tighness)):
        tighness[i] /= queries.shape[0]

    return tighness


NORMAL_PATH = "/vol/tmp/schaefpa/messi_datasets/"
SEISBENCH_PATH = "/vol/tmp/schaefpa/seismic/"

datasets = {
    # Other DS
    "ASTRO": ["astro.bin", "astro_queries.bin", 256, 0, np.float32],
    "BIGANN": ["bigANN.bin", "bigANN_queries.bin", 100, 0, np.int8],
    "SALD": ["SALD.bin", "SALD_queries.bin", 128, 0, np.float32],
    "SIFT1B": ["sift1b.bin", "sift1b_queries.bin", 128, 0, np.float32],
    "DEPP1B": ["deep1b.bin", "deep1b_queries.bin", 96, 0, np.float32],
    "SCEDC": ["SCEDC.bin", "SCEDC_queries.bin", 256, 0, np.float32],
    # Seisbench
    "ETHZ": ["ETHZ.bin", "ETHZ_queries.bin", 256, 1, np.float32],
    "ISC_EHB_DepthPhases": [
        "ISC_EHB_DepthPhases.bin",
        "ISC_EHB_DepthPhases_queries.bin",
        256,
        1,
        np.float32,
    ],
    "LenDB": ["LenDB.bin", "LenDB_queries.bin", 256, 1, np.float32],
    "Iquique": ["Iquique.bin", "Iquique_queries.bin", 256, 1, np.float32],
    "NEIC": ["NEIC.bin", "NEIC_queries.bin", 256, 1, np.float32],
    "OBS": ["OBS.bin", "OBS_queries.bin", 256, 1, np.float32],
    "OBST2024": ["OBST2024.bin", "OBST2024_queries.bin", 256, 1, np.float32],
    "PNW": ["PNW.bin", "PNW_queries.bin", 256, 1, np.float32],
    "Meier2019JGR": [
        "Meier2019JGR.bin",
        "Meier2019JGR_queries.bin",
        256,
        1,
        np.float32,
    ],
    "STEAD": ["STEAD.bin", "STEAD_queries.bin", 256, 1, np.float32],
    "TXED": ["TXED.bin", "TXED_queries.bin", 256, 1, np.float32],
}

all_threads = 128
n_segments = 16
alphabet_sizes = [256, 128, 64, 32, 16, 8, 4, 2]
all_csv_scores = {}
set_num_threads(all_threads)

for alphabet_size in alphabet_sizes:
    all_csv_scores[alphabet_size] = []

for alphabet_size in alphabet_sizes:
    for dataset in datasets:
        csv_scores = all_csv_scores[alphabet_size]

        # print("Running: ", dataset, alphabet_size)
        file_data, file_queries, d, path_switch, data_type = datasets[dataset]

        path = NORMAL_PATH if path_switch == 0 else SEISBENCH_PATH
        samples = read(
            path + file_data, dim=d, data_type=data_type, count=10_000_000
        )  # data
        queries = read(
            path + file_queries, data_type=data_type, dim=d, count=100
        )  # queries

        # print("\tDataset\t", dataset)
        # print("\tData Shape\t", samples.shape)
        # print("\tQuery Shape\t", queries.shape)

        # ignore all future warnings
        simplefilter(action="ignore", category=FutureWarning)
        simplefilter(action="ignore", category=UserWarning)

        # samples = samples[np.std(samples, axis=-1) > 1e-8]
        # queries = queries[np.std(queries, axis=-1) > 1e-8]

        # samples = (samples - np.mean(samples, axis=-1, keepdims=True)) / (
        #     np.std(samples, axis=-1, keepdims=True)
        # )
        # queries = (queries - np.mean(queries, axis=-1, keepdims=True)) / (
        #     np.std(queries, axis=-1, keepdims=True)
        # )

        np.nan_to_num(samples, nan=0, copy=False)
        np.nan_to_num(queries, nan=0, copy=False)

        SAX_transform = SAX(
            n_segments=n_segments, alphabet_size=alphabet_size, znormalized=True
        )
        SAX_samples = SAX_transform.fit_transform(samples).squeeze()
        PAA_queries = SAX_transform._get_paa(queries).squeeze()
        # print("\tSAX done.")

        histograms = ["equi-width", "equi-depth"]
        variances = [True, False]

        method_names = ["isax"]
        all_breakpoints = []
        all_dfts = []
        all_words = []

        for histogram, variance in itertools.product(histograms, variances):
            # sfa = SFAFast(
            #     word_length=n_segments,
            #     alphabet_size=alphabet_size,
            #     window_size=samples.shape[-1],
            #     binning_method=histogram,
            #     norm=True,
            #     variance=variance,
            #     lower_bounding_distances=True,
            #     n_jobs=all_threads,
            # )

            sfa = SFAWhole(
                word_length=n_segments,
                alphabet_size=alphabet_size,
                binning_method=histogram,
                variance=variance,
                norm=True,
                n_jobs=all_threads,
            )

            sfa.fit(samples)
            # X_dfts = sfa.transform_mft(queries).squeeze()
            # Y_words = sfa.transform_words(samples).squeeze()

            _, X_dfts = sfa.transform_words(queries)
            Y_words, _ = sfa.transform_words(samples)
            all_breakpoints.append(sfa.breakpoints.astype(np.float64))
            all_dfts.append(X_dfts.astype(np.float64))
            all_words.append(Y_words.astype(np.int32))

            # #print("Support", sfa.support)
            method_names.append(f"sfa_{histogram}_{variance}")

            # print(f"\tSFA {histogram} {variance} done.")

        sum_scores = {}
        for method_name in method_names:
            sum_scores[method_name] = {
                "dataset": [],
                "tightness": [],
            }

        # print("\tTransformation done. Computing Distances")

        tightness = compute_distances(
            queries,
            samples,
            PAA_queries,
            SAX_samples,
            SAX_transform.breakpoints,
            # sfa_transforms,
            np.array(all_breakpoints),
            np.array(all_dfts),
            np.array(all_words),
            np.array(method_names),
        )

        for i, method_name in enumerate(method_names):
            sum_scores[method_name]["dataset"].append(dataset)
            sum_scores[method_name]["tightness"].append(tightness[i])
            csv_scores.append((method_name, dataset, tightness[i]))

        # print(f"\n\n---- Results using {alphabet_size}-----")
        # for name, _ in sum_scores.items():
        #    print(
        #        f"---- Name {name}, \t tlb: {
        #          np.round(sum_scores[name]['tightness'], 3)}"
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
            f"logs/tlb_all_sofa_bench_{n_segments}_{alphabet_size}"
            f"-07-11-24-no_norm.csv",
            index=None,
        )
