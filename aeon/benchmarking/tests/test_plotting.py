import pandas as pd

from aeon.benchmarking.results_plotting import plot_scatter

import numpy as np

def test_plot_scatter():
    methods = ["1NN-DTW","MrSQM","RDST","RSF","STC"]

    path = "/Users/bzcschae/Downloads/aeon_results/ShapeletBased_Accuracy_mean.csv"

    df = pd.read_csv(path)


    ids = ["MrSQM", "STC"]
    plot = plot_scatter(
        df[ids].to_numpy(), ids[0], ids[1], figsize=(6, 6)
    )
    plot.show()

    plot.savefig("scatterplot.pdf")  # doctest: +SKIP