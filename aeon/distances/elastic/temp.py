from aeon.distances import dtw_distance
from aeon.testing.data_generation import make_example_2d_numpy_series

if __name__ == "__main__":
    x = make_example_2d_numpy_series(10, 1, random_state=1)
    y = make_example_2d_numpy_series(10, 1, random_state=2)

    dtw_distance(x, y)
