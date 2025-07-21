from itertools import product

import numpy as np
from numba import njit, prange
from numba.core import types
from numba.typed import Dict
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

from aeon.transformations.collection import BaseCollectionTransformer


class SPARTAN(BaseCollectionTransformer):
    """SPARTAN."""

    _tags = {
        "requires_y": False,
        "capability:multithreading": False,
        "algorithm_type": "dictionary",
    }

    def __init__(
        self,
        word_length=4,
        alphabet_size=8,  # this breaks flake8 [8, 4, 4, 2],
        window_size=0,
        binning_method="equi-depth",
        alphabet_allocation_method="DAA",  # direct or DAA
        learn_alphabet_lambda=0.5,
        remove_repeat_words=False,
        bit_budget=16,
        build_histogram=True,
        downsample=1.0,
        pca_solver="auto",
        dilation=0,
        first_difference=False,
        return_sparse=True,
        random_state=None,
    ):

        if isinstance(alphabet_size, int):
            self.alphabet_size = np.zeros(word_length, dtype=np.uint32)
            self.alphabet_size[:] = alphabet_size
        else:
            self.alphabet_size = np.array(alphabet_size, dtype=np.uint32)

        self.window_size = window_size

        self.word_length = word_length
        self.binning_method = binning_method
        self.remove_repeat_words = remove_repeat_words
        self.alphabet_allocation_method = alphabet_allocation_method
        self.bit_budget = bit_budget
        self.learn_alphabet_lambda = learn_alphabet_lambda
        self.build_histogram = build_histogram
        self.downsample = downsample
        self.pca_solver = pca_solver

        self.return_sparse = return_sparse
        self.dilation = dilation
        self.first_difference = first_difference

        # Feature selection part
        self.feature_count = 0
        self.relevant_features = None

        self.word_to_num = None
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y=None):
        X = X.squeeze(1)

        if self.dilation >= 1 or self.first_difference:
            X2, _ = _dilation(X, self.dilation, self.first_difference)
        else:
            X2, _ = X, np.arange(X.shape[-1])

        self.pca = PCA(n_components=self.word_length, svd_solver=self.pca_solver)
        self._X = X2
        self._y = y

        # do random (down)sampling if required
        if self.downsample < 1.0:
            sampling_num = min(max(int(np.ceil(len(X2) * self.downsample)), 10), 1000)
            random_indices = np.random.choice(X2.shape[0], sampling_num, replace=False)
            self._X_downsampled = self._X[random_indices]

        # check data statistics
        n_instances, series_length = X2.shape
        if self.window_size == 0:
            window_size = series_length
            self.window_size = window_size
        elif self.window_size < 1:
            window_size = int(series_length * self.window_size)
            self.window_size = window_size

        self.window_size = min(self.window_size, series_length)
        self.window_size = max(self.word_length, self.window_size)
        window_size = self.window_size

        # split data (for BOP)
        num_windows_per_inst = series_length - window_size + 1
        split = X2[
            :,
            np.arange(window_size)[None, :] + np.arange(num_windows_per_inst)[:, None],
        ]

        # --- Numeric Approximation ---
        if num_windows_per_inst == 1:
            if self.downsample < 1.0:
                # print("original shape: ", X.shape)
                # print("downsampled shape: ", self._X_downsampled.shape)

                self.pca.fit(self._X_downsampled)
                X_transform = self.pca.transform(X2)
            else:
                X_transform = self.pca.fit_transform(X2)
        elif self.window_size != 0 and self.window_size != series_length:

            flat_split = np.reshape(split, (split.shape[0], -1))
            if self.downsample < 1.0:
                self.pca.fit(flat_split[random_indices])
                split_transorm = self.pca.transform(flat_split)
            else:
                split_transorm = self.pca.fit_transform(flat_split)

        self.evcr = self.pca.explained_variance_ratio_

        # --- Discretization ---
        # alphabet allocation
        if self.alphabet_allocation_method == "direct":
            self.alphabet_size = self.alphabet_size
            self.avg_alphabet_size = int(np.mean(self.alphabet_size))
            # assigned_evc = self.evcr[0:self.word_length]
            # assigned_evc = assigned_evc / np.sum(assigned_evc)
        elif self.alphabet_allocation_method in ["DAA", "dp"]:
            if isinstance(self.alphabet_size, np.ndarray):
                alphabet_size = int(self.alphabet_size.mean())

            self.avg_alphabet_size = alphabet_size
            if self.bit_budget != int(np.log2(alphabet_size) * self.word_length):
                total_bit = self.bit_budget = int(
                    np.log2(alphabet_size) * self.word_length
                )
            else:
                total_bit = self.bit_budget

            # truncate and re-norm
            assigned_evc = self.evcr[0 : self.word_length]
            assigned_evc = assigned_evc / np.sum(assigned_evc)
            # avg_allocation = self.bit_budget // self.word_length

            DP_reward, bit_arr = _dynamic_alphabet_allocation(
                total_bit=total_bit, EV=assigned_evc, lamda=self.learn_alphabet_lambda
            )
            self.alphabet_size = np.array(
                [int(2 ** bit_arr[i]) for i in range(len(bit_arr))], dtype=np.uint32
            )

        # binning
        if num_windows_per_inst == 1:
            kept_components = X_transform[:, 0 : self.word_length]
            self.pca_repr = kept_components
            self.breakpoints = self.binning(kept_components)
            words = generate_words(X_transform, self.breakpoints, self.word_length)
        else:
            breakpoints = self.binning(split_transorm)
            self.pca_repr = split_transorm[:, 0 : self.word_length]
            self.breakpoints = breakpoints
            words = generate_words(split_transorm, self.breakpoints, self.word_length)

        self.transform_to_bag(words, self.word_length, y)

    def transform_words(self, X):
        if X.ndim == 3:
            X = X.squeeze(1)

        n_instances, series_length = X.shape

        self.window_size = min(self.window_size, X.shape[-1])
        window_size = self.window_size
        if self.window_size == 0:
            window_size = series_length

        num_windows_per_inst = series_length - window_size + 1

        if self.window_size == 0 or self.window_size == X.shape[1]:
            X_pca = self.pca.transform(X)
            words = generate_words(X_pca, self.breakpoints, self.word_length)
        else:
            split = X[:, (np.arange(window_size)[None, :]
                          + np.arange(num_windows_per_inst)[:, None]),]
            flat_split = np.reshape(split, (split.shape[0], -1))
            X_pca = self.pca.transform(flat_split)
            words = generate_words(X_pca, self.breakpoints, self.word_length)

        if self.build_histogram:
            new_words = words[:, None, :] if words.ndim == 2 else words
            self.pred_histogram = self.bag_to_hist_DAA(self.create_bags(new_words))
        else:
            self.pred_histogram = np.zeros((1, 1))

        return words.squeeze(), X_pca.squeeze()

    def _transform(self, X, y=None):
        X = X.squeeze(1)

        if self.dilation >= 1 or self.first_difference:
            X2, _ = _dilation(X, self.dilation, self.first_difference)
        else:
            X2, _ = X, np.arange(X.shape[-1])

        words, _ = self.transform_words(X2)

        if self.build_histogram:
            # transform: applies the feature selection strategy
            empty_dict = Dict.empty(
                key_type=types.uint32,
                value_type=types.uint32,
            )

            # transform
            bags = create_bag_transform(
                self.feature_count,
                self.relevant_features if self.relevant_features else empty_dict,
                words,
                self.remove_repeat_words,
            )[0]

            if self.return_sparse:
                bags = csr_matrix(bags, dtype=np.uint32)
            return bags
        else:
            return words

    def _fit_transform(self, X, y=None):
        self.fit(X, y=None)
        return self.transform(X)

    def binning(self, pca):
        if (self.binning_method == "equi-depth") or (
            self.binning_method == "equi-width"
        ):
            self.breakpoints = _mcb(
                pca, self.alphabet_size, self.word_length, self.binning_method
            )
            return self.breakpoints
        return None

    def transform_to_bag(self, words, word_len, y=None):
        """Transform words to bag-of-pattern and apply feature selection."""
        if (self.breakpoints.shape[1] <= 2) and (self.word_length <= 8):
            bag_of_words = create_bag_none(
                self.breakpoints,
                words.shape[0],
                words,
                word_len,
                self.remove_repeat_words,
            )
        else:
            feature_names = create_feature_names(words)
            feature_count = len(list(feature_names))
            relevant_features_idx = np.arange(feature_count, dtype=np.uint32)
            bag_of_words, self.relevant_features = create_bag_feature_selection(
                words.shape[0],
                relevant_features_idx,
                np.array(list(feature_names)),
                words,
                self.remove_repeat_words,
            )

        self.feature_count = bag_of_words.shape[1]
        if self.return_sparse:
            bag_of_words = csr_matrix(bag_of_words, dtype=np.uint32)

        return bag_of_words

    def create_bags(self, wordslists):
        n_instances, n_words_per_inst, word_length = wordslists.shape

        remove_repeat_words = self.remove_repeat_words
        wordslists = wordslists.astype(np.int32)
        bags = []
        last_word = None
        for i in range(n_instances):
            bag = {}
            wordlist = wordslists[i]
            for j in range(n_words_per_inst):
                word = wordlist[j]
                # print(word)
                text = "".join(map(str, word))
                if (not remove_repeat_words) or (text != last_word):
                    bag[text] = bag.get(text, 0) + 1

                last_word = text
            bags.append(bag)

        return bags

    def bag_to_hist(self, bags):
        n_instances = len(bags)

        word_length = self.word_length

        possible_words = self.alphabet_size[0] ** word_length
        # print("possible_words: ", possible_words)
        word_to_num = [
            np.base_repr(i, base=self.alphabet_size[0]) for i in range(possible_words)
        ]

        word_to_num = ["0" * (word_length - len(word)) + word for word in word_to_num]
        all_win_words = np.zeros((n_instances, possible_words))

        for j in range(n_instances):
            bag = bags[j]

            for key in bag.keys():
                v = bag[key]

                n = word_to_num.index(key)

                all_win_words[j, n] = v
        return all_win_words

    def combination_words_DAA(self, alphabet_sizes):
        """Generate possible words given alphabet sizes for each letter and word length.

        :param alphabet_sizes: List[int], List where each element specifies the number
               of choices for that letter position.
        :param word_length: int, Length of the word to be generated.
        :return: List[str], List of all possible words.
        """
        # Create a list of ranges based on the alphabet sizes
        ranges = [range(size) for size in alphabet_sizes]

        # Generate all possible combinations
        all_combinations = product(*ranges)

        # Convert each combination to a string and store in the list
        all_words = ["".join(map(str, combination)) for combination in all_combinations]

        return all_words

    def bag_to_hist_DAA(self, bags):
        n_instances = len(bags)

        word_length = self.word_length

        if isinstance(self.alphabet_size, (list, np.ndarray)):
            possible_words = int(np.prod(self.alphabet_size))
        else:
            possible_words = int(self.alphabet_size**word_length)

        if self.word_to_num is None:
            word_to_num = self.combination_words_DAA(self.alphabet_size)
            self.word_to_num = word_to_num

        all_win_words = np.zeros((n_instances, possible_words))

        for j in range(n_instances):
            bag = bags[j]
            for key in bag.keys():
                v = bag[key]
                n = self.word_to_num.index(key)
                all_win_words[j, n] = v

        return all_win_words


@njit(fastmath=True, cache=True)
def regularization_term(x, ev_value, avg_bit, lamda=0.5):
    return -lamda * (x - avg_bit) ** 2 * ev_value


@njit(fastmath=True, cache=True)
def trace_backwards(alloc, K, N):
    bit_arr2 = np.zeros(K, dtype=np.int32)
    unused_bit = N
    for i in range(K, 1, -1):
        bit_arr2[-i] = alloc[i, unused_bit]
        unused_bit -= alloc[i, unused_bit]
    bit_arr2[-1] = unused_bit
    return bit_arr2


@njit(cache=True, fastmath=True)
def _dynamic_alphabet_allocation(total_bit, EV, lamda=0.5):
    K = len(EV)
    N = total_bit
    A = int(N / K)
    DP = np.zeros((K + 1, N + 1))
    min_bit = 0
    max_bit = int(np.max(EV) * N)
    alloc = np.zeros_like(DP).astype(np.int32) + N

    # init
    DP[:, :] = -np.inf
    DP[0, 0] = 0

    # non-recursive
    for i in range(1, K + 1):
        for j in range(0, N + 1):
            max_reward = -np.inf
            for x in range(min_bit, max_bit + 1):
                if j - x >= 0 and x <= alloc[i - 1, j - x]:
                    current_reward = (
                        DP[i - 1, j - x] + x * EV[i - 1]
                        + regularization_term(x, EV[i - 1], A, lamda)
                    )

                    if current_reward > max_reward:
                        alloc[i, j] = x
                        max_reward = current_reward
                        DP[i, j] = current_reward

    bit_arr = trace_backwards(alloc, K, N)
    # print(np.sum(bit_arr), N)

    assert np.sum(bit_arr) == N
    return DP[K][N], bit_arr[::-1]


@njit(fastmath=True, cache=True)
def generate_words(pca, breakpoints, word_length):
    words = np.zeros((pca.shape[0], word_length), dtype=np.uint32)
    for a in range(pca.shape[0]):
        for i in range(word_length):
            words[a, i] = np.digitize(pca[a, i], breakpoints[i], right=True)

    return words


@njit(fastmath=True, cache=True, parallel=True)
def _mcb(pca, alphabet_size, word_length, binning_method):
    max_alphabet_size = np.max(alphabet_size)
    breakpoints = np.full(
        (word_length, max_alphabet_size), np.finfo(np.float32).max, dtype=np.float32
    )
    pca = np.round(pca, 4)

    for letter in prange(word_length):
        curr_alphabet_size = alphabet_size[letter]
        column = np.sort(pca[:, letter])
        bin_index = 0

        # use equi-depth binning
        if binning_method == "equi-depth":
            target_bin_depth = len(pca) / curr_alphabet_size

            for bp in range(curr_alphabet_size - 1):
                bin_index += target_bin_depth
                breakpoints[letter, bp] = column[int(bin_index)]

        # use equi-width binning aka equi-frequency binning
        elif binning_method == "equi-width":
            target_bin_width = (column[-1] - column[0]) / curr_alphabet_size

            for bp in range(curr_alphabet_size - 1):
                breakpoints[letter, bp] = (bp + 1) * target_bin_width + column[0]

    return breakpoints


def _dilation(X, d, first_difference):
    padding = np.zeros((len(X), 10))
    X = np.concatenate((padding, X, padding), axis=1)

    # using only first order differences
    if first_difference:
        X = np.diff(X, axis=1, prepend=0)

    # adding dilation
    X_dilated = _dilation2(X, d)
    X_index = _dilation2(
        np.arange(X_dilated.shape[-1], dtype=np.float64).reshape(1, -1), d
    )[0]

    return (
        X_dilated,
        X_index,
    )


@njit(cache=True, fastmath=True)
def _dilation2(X, d):
    # dilation on actual data
    if d > 1:
        start = 0
        data = np.zeros(X.shape, dtype=np.float64)
        for i in range(0, d):
            curr = X[:, i::d]
            end = curr.shape[1]
            data[:, start : start + end] = curr
            start += end
        return data
    else:
        return X.astype(np.float64)


@njit(cache=True, fastmath=True)
def create_feature_names(words):
    feature_names = set()
    for t_words in words:
        for t_word in t_words:
            feature_names.add(t_word)
    return feature_names


@njit(cache=True, fastmath=True)  # does not work with parallel=True ??
def create_bag_none(breakpoints, n_cases, words, word_length, remove_repeat_words):
    feature_count = np.uint32(breakpoints.shape[1] ** word_length)
    all_win_words = np.zeros((n_cases, feature_count), dtype=np.uint32)

    for j in prange(words.shape[0]):
        # this mask is used to encode the repeated words
        if remove_repeat_words:
            masked = np.nonzero(words[j])
            all_win_words[j, :] = np.bincount(words[j][masked], minlength=feature_count)
        else:
            all_win_words[j, :] = np.bincount(words[j], minlength=feature_count)

    return all_win_words


@njit(cache=True, fastmath=True)
def create_bag_feature_selection(
    n_cases,
    relevant_features_idx,
    feature_names,
    words,
    remove_repeat_words,
):
    relevant_features = Dict.empty(key_type=types.uint32, value_type=types.uint32)
    for k, v in zip(
        feature_names[relevant_features_idx],
        np.arange(len(relevant_features_idx), dtype=np.uint32),
    ):
        relevant_features[k] = v

    if remove_repeat_words:
        if 0 in relevant_features:
            del relevant_features[0]

    all_win_words = np.zeros((n_cases, len(relevant_features_idx)), dtype=np.uint32)
    for j in range(words.shape[0]):
        for key in words[j]:
            if key in relevant_features:
                all_win_words[j, relevant_features[key]] += 1
    return all_win_words, relevant_features


@njit(cache=True, fastmath=True, parallel=True)
def create_bag_transform(
    feature_count,
    relevant_features,
    words,
    remove_repeat_words,
):
    all_win_words = np.zeros((len(words), feature_count), np.uint32)
    for j in prange(words.shape[0]):
        if len(relevant_features) == 0:
            # this mask is used to encode the repeated words
            if remove_repeat_words:
                masked = np.nonzero(words[j])
                all_win_words[j, :] = np.bincount(
                    words[j][masked], minlength=feature_count
                )
            else:
                all_win_words[j, :] = np.bincount(words[j], minlength=feature_count)
        else:
            if remove_repeat_words:
                if 0 in relevant_features:
                    del relevant_features[0]

            for _, key in enumerate(words[j]):
                if key in relevant_features:
                    o = relevant_features[key]
                    all_win_words[j, o] += 1

    return all_win_words, all_win_words.shape[1]
