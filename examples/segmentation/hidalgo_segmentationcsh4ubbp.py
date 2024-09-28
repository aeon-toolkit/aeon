# %%NBQA-CELL-SEPfc780c
import numpy as np

from aeon.segmentation import HidalgoSegmenter

X = np.random.rand(100, 3)
X[:60, 1:] += 10
X[60:, 1:] = 0
hidalgo = HidalgoSegmenter(K=2, burn_in=0.8, n_iter=1000, seed=10)

hidalgo.fit_predict(X, axis=0)
