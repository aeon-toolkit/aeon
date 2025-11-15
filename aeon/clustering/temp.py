import numpy as np
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.clustering import TimeSeriesKMeans

if __name__ == "__main__":

    all_equal = []

    for i in range(10):
        for k in range(1, 10):
            n_cases = 10 * i + 10
            X = make_example_3d_numpy(n_cases, 1, n_cases, return_y=False)

            clst = TimeSeriesKMeans(n_clusters=k, random_state=1)
            clst._check_params(X)
            init = clst._kmeans_plus_plus_center_initializer_test(X)

            other_clst = TimeSeriesKMeans(n_clusters=k, random_state=1)
            other_clst._check_params(X)
            other_init = other_clst._kmeans_plus_plus_center_initializer(X)

            print(np.array_equal(init, other_init))
            all_equal.append(np.array_equal(init, other_init))

    print(all_equal)
    # Check if all values are True
    assert all(all_equal)
