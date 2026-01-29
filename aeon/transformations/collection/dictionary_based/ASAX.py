import ASAX_util
import numpy as np
import SAX
from numba import njit


@njit(cache=True, fastmath=True)
def MiSAX_ENTmN(timeSeries, nb_segments, alphabet_size, seg_lim):
    nb_timeSeries = timeSeries.shape[0]
    ts_len = timeSeries.shape[1]
    minn = np.min(timeSeries)
    maxx = np.max(timeSeries)
    cuts = Util.cuts_ENT(minn, maxx, alphabet_size)
    index = ts_len // 2
    indexes = np.array([0, index, ts_len])

    isaxRepresentationSaved = list()

    for i in range(len(timeSeries)):
        ts_PAA = Util.PAA_varSegSize(timeSeries[i], indexes)
        isaxRepresentationSaved.append(SAX.saxRep(ts_PAA, cuts))

    k = 2
    while k != nb_segments:
        indexSplitPosition = 0
        h = 0
        for i in range(k):
            newIndex = indexes[i] + ((indexes[i + 1] - indexes[i]) // 2)
            if (newIndex - indexes[i]) > seg_lim:
                indexes_temp = np.array((indexes[i], newIndex, indexes[i + 1]))
                table = {}

                for j in range(len(timeSeries)):
                    # paa
                    ts_PAA = Util.PAA_varSegSize(timeSeries[j], indexes_temp)
                    isaxWord = SAX.saxRep(ts_PAA, cuts)
                    isaxWord = np.concatenate(
                        (
                            np.append(isaxRepresentationSaved[j][:i], isaxWord),
                            isaxRepresentationSaved[j][i + 1 :],
                        )
                    )
                    isaxWordStr = SAX.toStrUsingChr(isaxWord)

                    # calcul occurence
                    if isaxWordStr in table:
                        table[isaxWordStr] += 1
                    else:
                        table[isaxWordStr] = 1

                occs = np.array(list(table.values()))
                # calcul entropie
                ent = Util.entropy(occs, nb_timeSeries)
                if ent > h:
                    h = ent
                    indexSplitPosition = i

        newIndex = indexes[indexSplitPosition] + (
            (indexes[indexSplitPosition + 1] - indexes[indexSplitPosition]) // 2
        )
        indexes_temp = np.array(
            (indexes[indexSplitPosition], newIndex, indexes[indexSplitPosition + 1])
        )
        indexes = np.concatenate(
            (
                np.append(indexes[: indexSplitPosition + 1], newIndex),
                indexes[indexSplitPosition + 1 :],
            )
        )

        for j in range(len(timeSeries)):
            ts_PAA = Util.PAA_varSegSize(timeSeries[j], indexes_temp)
            isaxWord = SAX.saxRep(ts_PAA, cuts)
            isaxRepresentationSaved[j] = np.concatenate(
                (
                    np.append(
                        isaxRepresentationSaved[j][:indexSplitPosition], isaxWord
                    ),
                    isaxRepresentationSaved[j][indexSplitPosition + 1 :],
                )
            )
        k = k + 1

    return indexes


def iSAXOcc(timeSeries, nb_segments, alphabet_size):
    table = {}
    cuts = SAX.getBreakPoints(alphabet_size)
    for ts in timeSeries:
        # paa
        ts_PAA = Util.PAA_fixedSegSize(ts, nb_segments)
        # print(ts_PAA)
        # isaxrep
        isaxWord = SAX.saxRep(ts_PAA, cuts)
        isaxWordStr = SAX.toStr(isaxWord)
        print(isaxWordStr)
        # calcul occurence
        if isaxWordStr in table:
            table[isaxWordStr] += 1
        else:
            table[isaxWordStr] = 1
    occs = np.array(list(table.values()))
    print(occs)
    return occs
