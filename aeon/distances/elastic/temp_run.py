import numpy as np

from aeon.datasets import load_gunpoint
from aeon.distances.elastic._ted import (
    ted_alignment_path,
    ted_cost_matrix,
    ted_distance,
)
from aeon.visualisation.distances._alignment_path import _plot_alignment, _plot_path

if __name__ == "__main__":
    # X, y = load_gunpoint(split="train")
    #
    # first = X[0]
    # second = X[1]
    x = np.array(
        [
            0.7553383207,
            0.4460987596,
            1.197682907,
            0.1714334808,
            0.5639929213,
            0.6891222874,
            1.793828873,
            0.06570866314,
            0.2877381702,
            1.633620422,
        ]
    )
    y = np.array(
        [
            -0.01765193577,
            -1.536784164,
            -0.1413292622,
            -0.7609346135,
            -0.1767363331,
            -2.192007072,
            -0.1933165696,
            -0.4648166839,
            -0.9444888843,
            -0.239523623,
        ]
    )

    dist_temp = ted_distance(x, y)
    cost_matrix = ted_cost_matrix(x, y)
    alignment_path = ted_alignment_path(x, y)
    print(f"Distance: {dist_temp}")
    print(f"Cost Matrix: {cost_matrix}")

    for experiment in [1, 2, 3, 4, 5]:
        dist_kwargs = {
            "experiment": experiment,
        }
        # plt_path = _plot_path(x, y, method="ted", title="ted warping", dist_kwargs=dist_kwargs)
        #
        # plt_path.show()
        plt_alignment = _plot_alignment(
            x, y, method="ted", title="ted", dist_kwargs=dist_kwargs
        )
        plt_alignment.show()

    # plt_path = _plot_path(x, y, method="adtw", title="adtw warping")
    #
    # plt_path.show()
    # plt_alignment = _plot_alignment(x, y, method="adtw",
    #                                 title="adtw")
    # plt_alignment.show()
