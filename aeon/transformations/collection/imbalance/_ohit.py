"""OHIT over sampling algorithm.

An adaptation of the oversampling method based on DRSNN clustering.

Original authors:
#          zhutuanfei
"""

from collections import OrderedDict

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.utils import check_random_state
from aeon.transformations.collection import BaseCollectionTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import ledoit_wolf


__all__ = ["OHIT"]


class OHIT(BaseCollectionTransformer):
    """
    Over-sampling using the Over-sampling based on (OHIT).

    This method is based on Density-Ratio Shared Nearest Neighbor (DRSNN) clustering to find high-density regions
    of minority class samples and generate synthetic samples within these clusters and Shrinkage estimation of
    large-dimensional covariance matrix

    DRSNN also contains three parameters(i.e.,drT ,k and kapa),it is capable of selecting the proper value for
    drT around 1.In addition,k and kapa can be set in a complementary way to avoid the merging and
    dissociation of clusters,that is,a large k with a relatively low kapa.
    Parameters
    ----------
    k : int, the nearest neighbor parameter in SNN similarity
        if None, set k = int(np.ceil(n ** 0.5 * 1.25)) where n is the number of minority samples
    kapa : int, the nearest neighbor parameter in defining density ratio
        if None, set kapa = int(np.ceil(n ** 0.5)) where n is the number of minority samples
    drT : float, default=0.9, the threshold of density ratio.
    distance : str or callable, default='euclidean'
        Distance metric to use for KNN in SNN similarity.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    """

    _tags = {
        "requires_y": True,
    }

    def __init__(self, k=None, kapa=None, drT=0.9, distance='euclidean' ,random_state=None):
        self.k = k
        self.kapa = kapa
        self.drT = drT
        self.distance = distance
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y=None):

        unique, counts = np.unique(y, return_counts=True)
        target_stats = dict(zip(unique, counts))
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items()
            if key != class_majority
        }
        self.sampling_strategy_ = OrderedDict(sorted(sampling_strategy.items()))

        return self

    def _transform(self, X, y=None):
        X = np.squeeze(X, axis=1)
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            if len(target_class_indices) == 1:
                X_new = np.tile(X[target_class_indices], (n_samples, 1))
                y_new = np.full(n_samples, fill_value=class_sample, dtype=y.dtype)
                X_resampled.append(X_new)
                y_resampled.append(y_new)
                continue
            X_class = X[target_class_indices]
            n, m = X_class.shape
            # set the default value of k and kapa
            if self.k is None:
                self.k = int(np.ceil(n ** 0.5 * 1.25))
            if self.kapa is None:
                self.kapa = int(np.ceil(n ** 0.5))

            # Initialize NearestNeighbors for SNN similarity
            self.NearestNeighbors = NearestNeighbors(metric=self.distance, n_neighbors=self.k + 1)

            clusters, cluster_label = self._cluster_minority(X_class)
            Me, eigen_matrices, eigen_values = self._covStruct(X_class, clusters)

            # allocate the number of synthetic samples to be generated for each cluster
            random_state = check_random_state(self.random_state)
            os_ind = np.tile(np.arange(0, n), int(np.floor(n_samples / n)))
            remaining = random_state.choice(np.arange(0, n), n_samples - n * int(np.floor(n_samples / n)), replace=False)
            os_ind = np.concatenate([os_ind, remaining])
            R = 1.25 if len(clusters) > 1 else 1.1

            """generate  the structure-preserving synthetic samples for each cluster"""
            X_new = np.zeros((n_samples, m))
            count = 0
            # consider the samples in the cluster with label 0 i.e. the samples that are not clustered
            X_class_0 = X_class[cluster_label == 0]
            if X_class_0.size != 0:
                gen_0 = np.sum(np.isin(os_ind, np.where(cluster_label == 0)[0]))
                idx_0 = random_state.choice(len(X_class_0), gen_0, replace=True)
                X_new[count:count + gen_0, :] = X_class_0[idx_0]
                count += gen_0
            for i, cluster in enumerate(clusters):
                gen_i = np.sum(np.isin(os_ind, np.where(cluster_label == (i + 1))[0]))
                X_new[count:count + gen_i, :] = self._generate_synthetic_samples(
                                                Me[i], eigen_matrices[i], eigen_values[i], gen_i, R)
                count += gen_i

            assert count == n_samples
            X_resampled.append(X_new)
            y_new = np.full(n_samples, fill_value=class_sample, dtype=y.dtype)
            y_resampled.append(y_new)

        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)
        X_resampled = X_resampled[:, np.newaxis, :]
        return X_resampled, y_resampled

    def _cluster_minority(self, X):
        """Apply DRSNN clustering on minority class samples."""
        n = X.shape[0]
        k = self.k
        kapa = self.kapa
        drT = self.drT

        self.NearestNeighbors.fit(X)
        neighbors = self.NearestNeighbors.kneighbors(X, return_distance=False)[:,1:]
        """ construct the shared nearest neighbor similarity """
        strength = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                shared_nn = np.intersect1d(neighbors[i, :k], neighbors[j, :k])
                strength[i, j] = strength[j, i] = np.sum((k + 1 - np.searchsorted(neighbors[i, :k], shared_nn)) *
                                                   (k + 1 - np.searchsorted(neighbors[j, :k], shared_nn)))

        """ construct the shared nearest neighbor graph """
        strength_nn = np.sort(strength, axis=1)[:, ::-1][:, :k]
        idx_nn = np.argsort(strength, axis=1)[:, ::-1]
        graph = np.zeros((n, k))
        for i in range(n):
            for j in range(k):
                if np.any(idx_nn[idx_nn[i, j], :k] == i):
                    graph[i, j] = 1

        density = np.sum(strength_nn * graph, axis=1)
        density_ratio = np.zeros(n)
        for i in range(n):
            non_noise = np.where(density[idx_nn[i, :kapa]] != 0)[0]
            if non_noise.size == 0:
                density_ratio[i] = 0
            else:
                density_ratio[i] = density[i] / np.mean(density[idx_nn[i, non_noise]])

        """ identify core points """
        core_idx = np.where(density_ratio > drT)[0]
        """ find directly density-reachable samples for each core point"""
        neighborhood = {core: set(idx_nn[core, :kapa]) for core in core_idx}
        for i in core_idx:
            for j in core_idx:
                if np.any(idx_nn[j, :kapa] == i):
                    neighborhood[i].add(j)
        neighborhood = {key: list(value) for key, value in neighborhood.items()}

        clusters = []
        cluster_label = np.zeros(len(neighbors), dtype=int)
        cluster_id = 0

        for i in core_idx:
            if cluster_label[i] == 0:
                cluster_id += 1
                seed = [i]
                clusters.append(set(seed))
                while seed:
                    point = seed.pop(0)
                    idx = np.where(core_idx == point)[0]
                    if idx.size > 0 and cluster_label[point] == 0:
                        seed.extend(neighborhood[point])
                        clusters[-1].update(neighborhood[point])
                    cluster_label[point] = cluster_id
        # no cluster has been found, the whole samples are taken as one cluster
        if len(clusters) == 0:
            clusters.append(list(range(n)))
            cluster_label = np.ones(n, dtype=int)
        return clusters, cluster_label

    def _covStruct(self, data, clusters):
        """
        Calculate the covariance matrix of the minority samples.
        """
        Me, Eigen_matrices, Eigen_values = [], [], []
        for cluster in clusters:
            cluster = list(cluster)
            cluster_data = data[cluster]
            sigma, shrinkage = ledoit_wolf(cluster_data)
            me = np.mean(cluster_data, axis=0)
            eigenValues, eigenVectors = np.linalg.eigh(sigma)
            eigenValues = np.diag((eigenValues))
            Me.append(me)
            Eigen_matrices.append(eigenVectors)
            Eigen_values.append(eigenValues)
        return Me, Eigen_matrices, Eigen_values

    def _generate_synthetic_samples(self, Me, eigenMatrix, eigenValue, eta, R):
        """Generate synthetic samples based on clustered minority samples."""
        # Initialize the output sample generator and probability arrays
        n_samples = int(np.ceil(eta * R))
        SampGen = np.zeros((n_samples, len(Me)))
        Prob = np.zeros(n_samples)

        # Calculate the square root of the absolute eigenvalues
        DD = np.sqrt(np.abs(np.diag(eigenValue)))
        DD = DD.reshape(1, -1)

        # Initialize mean and covariance for the multivariate normal distribution
        Mu = np.zeros(len(Me))
        Sigma = np.eye(len(Me))

        for cnt in range(n_samples):
            # Generate a sample from the multivariate normal distribution
            S = np.random.multivariate_normal(Mu, Sigma, 1)
            Prob[cnt] = multivariate_normal.pdf(S, Mu, Sigma)

            # Scale the sample with the eigenvalues
            S = S * DD
            # Generate the final sample by applying the eigenvector matrix
            x = S @ eigenMatrix.T + Me
            SampGen[cnt, :] = x

        # Sort the samples based on the probability in descending order
        sorted_indices = np.argsort(Prob)[::-1]
        SampGen = SampGen[sorted_indices[:eta], :]

        return SampGen

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {"n_clusters": 3}