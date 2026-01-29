import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def means(ts, start, end):
    if (end - start > 0) and (end > start):
        return np.mean(ts[start:end])
    else:
        print("Error! start = ", start, " end = ", end)
        return 0


@njit(cache=True, fastmath=True)
def transformPAA(ts, n_segments):
    w = len(ts) // n_segments  # round to the closest int
    ts_PAA = np.zeros(n_segments, dtype=np.float64)  # PAA/Means
    ts_TREND = np.zeros(n_segments * 2, dtype=np.float64)  # Trend

    offset = 0
    for t in range(n_segments):
        tt = ts[offset : offset + int(w)]
        ts_PAA[t] = np.mean(tt)

        # Defined by Eq (4) in the paper
        ts_TREND[2 * t] = tt[0] - ts_PAA[t]

        # The proposed approach to re-use the last value of the preceeding segment
        # violates the lower bounding lemma!
        ts_TREND[2 * t + 1] = tt[-1] - ts_PAA[t]
        offset += int(w)

    return ts_PAA, ts_TREND


@njit(cache=True, fastmath=True)
def SAX_TD_transform(samples, n_segments, breakpoints):
    features = n_segments // 2  # we need two to three (!) features per segment

    # ts_PAAs = np.zeros((len(samples), w), dtype=np.float64)
    ts_SAX = np.zeros((len(samples), features), dtype=np.int32)
    ts_TREND = np.zeros((len(samples), 2 * features), dtype=np.float64)

    for j, sample in enumerate(samples):
        ts_PAA, ts_TREND[j] = transformPAA(sample, features)
        ts_SAX[j] = np.digitize(x=ts_PAA, bins=breakpoints)

    return ts_SAX, ts_TREND


@njit(cache=True)
def SAX_TD_MINDIST(q_s, q_trends, c_s, c_trends, seg_len, breakpoints):

    sum = np.float64(0.0)
    for i in range(len(q_s)):
        if abs(q_s[i] - c_s[i]) <= 1:
            continue
        else:
            dist = (
                breakpoints[max(q_s[i], c_s[i]) - 1] - breakpoints[min(q_s[i], c_s[i])]
            )
            sum += (dist**2) * seg_len

            # Non-Symbolic Part!! See Eq. (4) in the paper
            sum += (q_trends[2 * i] - c_trends[2 * i]) ** 2 + (
                q_trends[2 * i + 1] - c_trends[2 * i + 1]
            ) ** 2

    return np.sqrt(sum)
