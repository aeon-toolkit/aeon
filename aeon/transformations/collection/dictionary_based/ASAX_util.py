import csv
import random
from functools import cache

import ASAX
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def seg_mean(ts, start, end):
    if (end - start + 1 > 0) and (end > start):
        return np.mean(ts[start : end + 1])
    else:
        print("Error! start = ", start, " end = ", end)
        return 0


def segs_mean(timeSeries):
    nb_segments = timeSeries.shape[1]
    mean = np.empty(nb_segments)
    for i in range(nb_segments):
        mean[i] = np.mean(timeSeries[:, i])
    return mean


@njit(cache=True, fastmath=True)
def stdv(timeSeries):
    nb_segments = timeSeries.shape[1]
    stdev = np.empty(nb_segments)
    for i in range(nb_segments):
        stdev[i] = np.std(timeSeries[:, i])
    return stdev


def eucDistance(q, c):
    return np.sqrt(np.sum((q - c) ** 2))


def DR(q_PAA, c_PAA, ts_len):
    return np.sqrt(ts_len / len(q_PAA)) * np.sqrt(np.sum((q_PAA - c_PAA) ** 2))


@njit(cache=True, fastmath=True)
def DR_VAR(q_PAA, c_PAA, segs_len):
    sum = 0
    for i in range(len(q_PAA)):
        sum += ((q_PAA[i] - c_PAA[i]) ** 2) * (segs_len[i])

    return np.sqrt(sum)


def MINDIST(q_s, c_s, ts_len, cuts):
    return np.sqrt(ts_len / len(q_s)) * np.sqrt(sum_dist(q_s, c_s, cuts))


def sum_dist(q_s, c_s, cuts):
    sum = 0
    for i in range(len(q_s)):
        sum += (dist(q_s[i], c_s[i], cuts)) ** 2
    return sum


# # @njit(cache=True, fastmath=True)
# def dist2(r, r_paa, c, c_paa, cuts):
#     r = int(r)
#     c = int(c)
#     if abs(r - c) <= 1:
#         return 0
#
#     if c >= cuts.shape[0]:
#         br_upper = np.inf
#     else:
#         br_upper = cuts[c]
#
#     if c - 1 < 0:
#         br_lower = -np.inf
#     else:
#         br_lower = cuts[c - 1]
#
#     if c_paa > br_upper or c_paa < br_lower:
#         print ("Error! c_paa = ", c_paa, " br_upper = ", br_upper, " br_lower = ", br_lower)
#     #br_upper = cuts[c]
#     #br_lower = cuts[c-1]
#
#     if br_upper > r_paa:
#         return br_upper - r_paa
#     else:
#         return r_paa - br_lower
#
#
# # @njit(cache=True, fastmath=True)
# def MINDIST_VAR2(q_s, q_paas, c_s, c_paas, segs_len, cuts):
#     return np.sqrt(sum_dist_var2(q_s, q_paas, c_s, c_paas, segs_len, cuts))
#
#
# # @njit(cache=True, fastmath=True)
# def sum_dist_var2(q_s, q_paas, c_s, c_paas, segs_len, cuts):
#     sum = 0
#     for i in range(len(q_s)):
#         sum += ((dist2(q_s[i], q_paas[i], c_s[i], c_paas[i], cuts)) ** 2) * (segs_len[i])
#     return sum


@njit(cache=True, fastmath=True)
def dist(r, c, cuts):
    if abs(r - c) <= 1:
        return np.float64(0.0)
    else:
        return np.float64(cuts[max(r, c) - 1] - cuts[min(r, c)])


@njit(cache=True)
def MINDIST_VAR(q_s, c_s, segs_len, cuts):
    sum = np.float64(0.0)
    for i in range(len(q_s)):
        if abs(q_s[i] - c_s[i]) <= 1:
            continue
        else:
            dist = cuts[max(q_s[i], c_s[i]) - 1] - cuts[min(q_s[i], c_s[i])]
            sum += (dist**2) * segs_len[i]

    return np.sqrt(sum)


def segments_len(indexes):
    w = len(indexes) - 1
    segs_len = np.empty(w, dtype=int)
    for i in range(w):
        segs_len[i] = indexes[i + 1] - indexes[i]
    return segs_len


def PAA_fixedSegSize(ts, nb_segments):
    ts_len = len(ts)
    segment_size = ts_len / nb_segments
    ts_PAA = np.empty(nb_segments, dtype=np.float64)
    if nb_segments != ts_len:
        offset = 0
        for i in range(nb_segments):
            ts_PAA[i] = seg_mean(ts, offset, offset + int(segment_size) - 1)
            # print("paa i ",ts_PAA[i])
            offset += int(segment_size)
    else:
        ts_PAA = np.copy(ts)

    return ts_PAA


@njit(cache=True, fastmath=True)
def PAA_varSegSize(ts, indexes):
    nb_segments = len(indexes) - 1
    ts_PAA = np.zeros(nb_segments, dtype=np.float64)

    for i in range(nb_segments):
        ts_PAA[i] = seg_mean(ts, indexes[i], indexes[i + 1] - 1)

    return ts_PAA


@njit(cache=True, fastmath=True)
def entropy(occs, db_size):
    h = 0
    p = occs / db_size
    logp = np.log2(p)
    for i in range(len(occs)):
        h = np.sum(p * logp)
    return -h


def ds_entropyiSAX(timeSeries, nb_segments, alphabet_size):
    return entropy(
        ASAX.iSAXOcc(timeSeries, nb_segments, alphabet_size), timeSeries.shape[0]
    )


def ds_entropyMiSAX(timeSeries, nb_segments, alphabet_size):
    occ = ASAX.MiSAXOcc(timeSeries, nb_segments, alphabet_size)
    return entropy(occ, timeSeries.shape[0])


def readDataset(file, n, m):
    # nb line 40 million
    # nb column 200
    file = open("Example/" + file + ".txt")
    timeSeries = np.empty((n, m))
    for i in range(n):
        ts_StrValues = file.readline().split(",")
        # print(ts_StrValues)
        # print(i,"   ",len(ts_StrValues))

        for j in range(m):
            timeSeries[i, j] = float(ts_StrValues[j + 1])
    file.close()
    return timeSeries


def lenNN(NN):
    nb = 0
    for k in NN:
        nb += len(NN[k])
    return nb


def toRemove(NNv, query):
    i = 0
    max = eucDistance(query, NNv[0][1])
    for j in range(1, len(NNv)):
        dist = eucDistance(query, NNv[j][1])
        if dist > max:
            max = dist
            i = j
    return i


def StrToTS(line):
    tsStr = line.split(",")
    ts = np.empty(len(tsStr), dtype=float)
    for j in range(len(tsStr)):
        ts[j] = np.float32(tsStr[j])
    return ts


def queryFileToTS(path):
    timeSeries = list()
    file = open(path)
    line = file.readline()
    while line != "":
        # print(line)
        ts = StrToTS(line)
        timeSeries.append(ts)
        line = file.readline()
    return timeSeries


def accuracyPC(id_GT, id_App_KNN, K_NN):
    total = 0
    for id in id_App_KNN:
        if id in id_GT:
            total += 1

    return total / K_NN


def App_KNN_Search(NN_values, query):
    newList = list(NN_values)
    newList.sort(key=lambda x: eucDistance(query, x[1]))

    id_App_KNN = list()
    for e in newList:
        id_App_KNN.append(e[0])
    return id_App_KNN


def GT_KNN_Search(timeSeries, query, nb_NN):
    NN = dict()
    id = 0
    for ts in timeSeries:
        dist = eucDistance(query, ts)
        if dist in NN.keys():
            raise Exception("double")
        if len(NN) < nb_NN:
            NN[dist] = (id, ts)
        else:
            maxDist = max(NN)
            if dist < maxDist:
                del NN[maxDist]
                NN[dist] = (id, ts)
        id += 1

    newList = list(NN.values())

    newList.sort(key=lambda x: eucDistance(query, x[1]))

    id_GT_KNN = list()
    for e in newList:
        id_GT_KNN.append(e[0])
    return id_GT_KNN


def tsvToTxt(file, out):
    lines_seen = set()  # holds lines already seen
    outfile = open("Example/" + out + ".txt", "w")
    tsv_file = open(file)

    read_tsv = csv.reader(tsv_file, delimiter="\t")
    for row in read_tsv:
        line = str(row[0])
        for i in range(1, len(row)):
            line += "," + str(row[i])
        if line not in lines_seen:  # not a duplicate
            outfile.write(line + "\n")
            lines_seen.add(line)
    outfile.close()


@njit(cache=True, fastmath=True)
def cuts_ENT(min, max, card):
    offset = (max - min) / card
    cuts = np.empty(card - 1)
    cuts[0] = min + offset
    for i in range(1, len(cuts)):
        cuts[i] = cuts[i - 1] + offset
    return cuts
