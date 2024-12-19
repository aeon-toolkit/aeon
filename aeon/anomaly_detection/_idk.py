"""IDK² and s-IDK² anomaly detector."""

import random
import numpy as np
from aeon.anomaly_detection.base import BaseAnomalyDetector

class IDK(BaseAnomalyDetector):
    """IDK² and s-IDK² anomaly detector.

       The Isolation Distributional Kernel (IDK) is a data-dependent kernel for efficient
       anomaly detection, improving accuracy without explicit learning. Its extension,
       IDK², simplifies group anomaly detection, outperforming traditional methods in
       speed and effectiveness.

       .. list-table:: Capabilities
          :stub-columns: 1

          * - Input data format
            - univariate
          * - Output data format
            - anomaly scores
          * - Learning Type
            - unsupervised

       Parameters
       ----------
       psi1 : int
            Number of samples randomly selected in each iteration for the feature map matrix.
       psi2 : int
            Number of samples used for the second-stage feature map embedding.
       width : int
            Size of the sliding or fixed-width window for anomaly detection.
       t : int, default=100
            Number of iterations (time steps) for random sampling to construct feature maps.
       sliding : bool, default=False
            Whether to use a sliding window approach. If True, computes scores for sliding windows;
            otherwise, processes fixed-width segments.
       Notes
       -----
       This implementation is inspired by the Isolation Distributional Kernel (IDK)
       approach as detailed in [1]_.
       The code is adapted from the open-source repository [2]_.

       References
       ----------
       [1]Isolation Distributional Kernel: A New Tool for Kernel-Based Anomaly Detection.
            DOI: https://dl.acm.org/doi/10.1145/3394486.3403062

       [2] GitHub Repository:
            IsolationKernel/Codes: IDK Implementation for Time Series Data
            URL: https://github.com/IsolationKernel/Codes/tree/main/IDK/TS
        """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:missing_values": False,
    }
    def __init__(
            self,
            psi1,
            psi2,
            width,
            t=100,
            sliding = False,
    ):
        self.psi1 = psi1
        self.psi2 = psi2
        self.width = width
        self.t = t
        self.sliding  = sliding
        super().__init__(axis=0)

    def __IK_inne_fm(self,X, psi, t=100):
        onepoint_matrix = np.zeros((X.shape[0], (int)(t * psi)), dtype=int)
        for time in range(t):
            sample_num = psi  #
            sample_list = [p for p in range(len(X))]
            sample_list = random.sample(sample_list, sample_num)
            sample = X[sample_list, :]

            tem1 = np.dot(np.square(X), np.ones(sample.T.shape))  # n*psi
            tem2 = np.dot(np.ones(X.shape), np.square(sample.T))
            point2sample = tem1 + tem2 - 2 * np.dot(X, sample.T)  # n*psi

            sample2sample = point2sample[sample_list, :]
            row, col = np.diag_indices_from(sample2sample)
            sample2sample[row, col] = 99999999
            radius_list = np.min(sample2sample, axis=1)

            min_point2sample_index = np.argmin(point2sample, axis=1)
            min_dist_point2sample = min_point2sample_index + time * psi
            point2sample_value = point2sample[range(len(onepoint_matrix)), min_point2sample_index]
            ind = point2sample_value < radius_list[min_point2sample_index]
            onepoint_matrix[ind, min_dist_point2sample[ind]] = 1

        return onepoint_matrix

    def __IDK(self,X, psi, t=100):
        point_fm_list = self.__IK_inne_fm(X=X, psi=psi, t=t)
        feature_mean_map = np.mean(point_fm_list, axis=0)
        return np.dot(point_fm_list, feature_mean_map) / t

    def _IDK_T(self,X):
        window_num = int(np.ceil(X.shape[0] / self.width))
        featuremap_count = np.zeros((window_num, self.t *self.psi1))
        onepoint_matrix = np.full((X.shape[0], self.t), -1)

        for time in range(self.t):
            sample_num = self.psi1
            sample_list = [p for p in range(X.shape[0])]
            sample_list = random.sample(sample_list, sample_num)
            sample = X[sample_list, :]
            tem1 = np.dot(np.square(X), np.ones(sample.T.shape))  # n*psi
            tem2 = np.dot(np.ones(X.shape), np.square(sample.T))
            point2sample = tem1 + tem2 - 2 * np.dot(X, sample.T)  # n*psi

            sample2sample = point2sample[sample_list, :]
            row, col = np.diag_indices_from(sample2sample)
            sample2sample[row, col] = 99999999

            radius_list = np.min(sample2sample, axis=1)
            min_dist_point2sample = np.argmin(point2sample, axis=1)  # index

            for i in range(X.shape[0]):
                if point2sample[i][min_dist_point2sample[i]] < radius_list[min_dist_point2sample[i]]:
                    onepoint_matrix[i][time] = min_dist_point2sample[i] + time * self.psi1
                    featuremap_count[(int)(i / self.width)][onepoint_matrix[i][time]] += 1


        for i in range((int)(X.shape[0] / self.width)):
            featuremap_count[i] /= self.width
        isextra = X.shape[0] - (int)(X.shape[0] / self.width) * self.width
        if isextra > 0:
            featuremap_count[-1] /= isextra

        if isextra > 0:
            featuremap_count = np.delete(featuremap_count, [featuremap_count.shape[0] - 1], axis=0)

        return self.__IDK(featuremap_count, psi=self.psi2, t=self.t)
    def _IDK_square_sliding(self,X):
        point_fm_list = self.__IK_inne_fm(X=X, psi=self.psi1, t=self.t)
        point_fm_list=np.insert(point_fm_list, 0, 0, axis=0)
        cumsum=np.cumsum(point_fm_list,axis=0)

        subsequence_fm_list=(cumsum[self.width:]-cumsum[:-self.width])/float(self.width)

        return self.__IDK(X=subsequence_fm_list, psi=self.psi2, t=self.t)
    def _predict(self,X):
        if self.sliding:
            return self._IDK_square_sliding(X)
        return self._IDK_T(X)