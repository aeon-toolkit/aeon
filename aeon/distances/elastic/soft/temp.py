import numpy as np

from aeon.datasets import load_from_ts_file, load_japanese_vowels
from aeon.distances import distance, euclidean_distance, squared_distance

if __name__ == "__main__":
    # temp = load_japanese_vowels(split="test")
    DATASET_NAME = "GunPoint"
    temp, labs = load_from_ts_file(
        f"/Users/chrisholder/Documents/Research/datasets/UCR/Univariate_ts/{DATASET_NAME}/{DATASET_NAME}_TRAIN.ts"
    )
    # temp1 = np.array([[1, 2, 3], [4, 5, 6]])
    # temp2 = np.array([[10, 12, 13], [14, 15, 16]])

    # test = squared_distance(temp1, temp2)

    temp = distance(temp[0], temp[1], method="dtw")

    # find the longest series
    longest = 0
    i = 0
    for series in temp[0]:
        for channel in series:
            # If array contains a nan
            if np.isnan(channel).any():
                print(f"{i} Contains NaN")
        i += 1
    stop = ""
