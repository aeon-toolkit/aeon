"""Shapelet plotting tools."""

__maintainer__ = []

__all__ = []


import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from aeon.classification.shapelet_based._ls import LearningShapeletClassifier
from aeon.classification.shapelet_based._rdst import RDSTClassifier
from aeon.classification.shapelet_based._rsast_classifier import RSASTClassifier
from aeon.classification.shapelet_based._sast_classifier import SASTClassifier
from aeon.classification.shapelet_based._stc import ShapeletTransformClassifier
from aeon.transformations.collection.shapelet_based._dilated_shapelet_transform import (
    RandomDilatedShapeletTransform,
    compute_shapelet_dist_vector,
    get_all_subsequences,
    normalize_subsequences,
)
from aeon.transformations.collection.shapelet_based._rsast import RSAST
from aeon.transformations.collection.shapelet_based._sast import SAST
from aeon.transformations.collection.shapelet_based._shapelet_transform import (
    RandomShapeletTransform,
)
from aeon.utils.numba.general import sliding_mean_std_one_series


class _Shapelet:
    """
    A Shapelet object to use for ploting operations.

    Parameters
    ----------
    values : array, shape=(n_channels, length)
        Values of the shapelet.
    length : int
        Length of the shapelet.
    normalize : bool
        Wheter the shapelet use a normalized distance.
    dilation : int
        Dilation of the shapelet. The default is 1, which is equivalent to no
        dilation.
    threshold : float
        Lambda threshold for Shapelet Occurrence feature. The default value is None
        if it is not used (used in RDST).
    """

    def __init__(
        self,
        values,
        length,
        normalize,
        dilation=1,
        threshold=None,
    ):
        self.values = np.asarray(values)
        self.length = length
        self.normalize = normalize
        self.threshold = threshold
        self.dilation = dilation

    def plot(self, figsize=(10, 5), seaborn_context="talk", ax=None):
        """
        Plot the shapelet values.

        Parameters
        ----------
        figsize : tuple, optional
            2D size of the figure. The default is (10,5).
        seaborn_context : str, optional
            Seaborn module context. The default is 'talk'.
        ax : matplotlib axe, optional
            A matplotlib axe on which to plot the figure. The default is None
            and will create a new figure of size figsize.

        Returns
        -------
        fig : matplotlib figure
            The resulting figure
        """
        title_string = ""
        if self.dilation > 1:
            title_string += f"dil={self.dilation}"
        if self.threshold is not None:
            title_string += f"threshold={np.round(self.threshold, decimals=2)}"
        title_string += f"normalize={self.normalize}"

        if ax is None:
            sns.set()
            sns.set_context(seaborn_context)
            fig = plt.figure(figsize=(figsize))
            plt.plot(self.values)
            plt.title(title_string)
            return fig
        else:
            ax.plot(self.values)
            ax.set_title(title_string)
            return ax

    def plot_on_X(
        self,
        X,
        figsize=(10, 5),
        seaborn_context="talk",
        alpha=0.9,
        shp_dot_size=40,
        shp_c="purple",
        ax=None,
        label=None,
        x_linewidth=2,
    ):
        """
        Plot the shapelet on its best match on the time series X.

        Parameters
        ----------
        X : array, shape=(n_features, n_timestamps)
            Input time series
        figsize : tuple, optional
            Size of the figure. The default is (10,5).
        seaborn_context : str, optional
            Seaborn context. The default is 'talk'.
        alpha : float, optional
            Alpha parameter for plotting X. The default is 0.9.
        shp_dot_size : float, optional
            Size of the scatter plot to represent the shapelet on X.
            The default is 40.
        shp_c : str, optional
            Color of the shapelet scatter plot. The default is 'purple'.
        ax : matplotlib axe, optional
            A matplotlib axe on which to plot the figure. The default is None
            and will create a new figure of size figsize.
        label : str, optional
            Custom label to plot as legend for X. The default is None.
        x_linewidth : float, optional
            The linewidth of X plot. The default is 2.

        Returns
        -------
        fig : matplotlib figure
            The resulting figure with S on its best match on X. A normalized
            shapelet will be scalled to macth the scale of X.

        """
        if len(X.shape) == 1:
            X = X[np.newaxis, :]
        # Get candidate subsequences in X
        X_subs = get_all_subsequences(X, self.length, self.dilation)
        # Normalize candidates and shapelet values
        if self.normalize:
            X_means, X_stds = sliding_mean_std_one_series(X, self.length, self.dilation)
            X_subs = normalize_subsequences(X_subs, X_means, X_stds)
            _values = (self.values - self.values.mean(axis=-1)) / self.values.std(
                axis=1
            )
        else:
            _values = self.values

        # Compute distance vector
        c = compute_shapelet_dist_vector(X, _values, self.length)

        # Get best match index
        idx_match = np.asarray(
            [(c.argmin() + i * self.dilation) % X.shape[0] for i in range(self.length)]
        ).astype(int)

        # If normalize, scale back the values of the shapelet to the scale of the match
        if self.norm:
            _values = (_values * X[idx_match].std()) + X[idx_match].mean()

        if ax is None:
            sns.set()
            sns.set_context(seaborn_context)
            fig = plt.figure(figsize=(figsize))
            plt.plot(X, label=label, linewidth=x_linewidth, alpha=alpha)
            plt.scatter(
                idx_match, _values, s=shp_dot_size, c=shp_c, zorder=3, alpha=alpha
            )
            return fig
        else:
            ax.plot(X, label=label, linewidth=x_linewidth, alpha=alpha)
            ax.scatter(
                idx_match, _values, s=shp_dot_size, c=shp_c, zorder=3, alpha=alpha
            )

    def plot_distance_vector(
        self,
        X,
        figsize=(10, 5),
        seaborn_context="talk",
        c_threshold="purple",
        ax=None,
        label=None,
    ):
        """
        Plot the shapelet distance vector computed between itself and X.

        Parameters
        ----------
        X : array, shape=(n_features, n_timestamps)
            Input time series
        figsize : tuple, optional
            Size of the figure. The default is (10,5).
        seaborn_context : str, optional
            Seaborn context. The default is 'talk'.
        c_threshold : float, optional
            Color used to represent a line on the y-axis to visualize the lambda
            threshold. The default is 'purple'.
        ax : matplotlib axe, optional
            A matplotlib axe on which to plot the figure. The default is None
            and will create a new figure of size figsize.
        label : str, optional
            Custom label to plot as legend. The default is None.

        Returns
        -------
        fig : matplotlib figure
            The resulting figure with the distance vector obtained by d(S,X)

        """
        if len(X.shape) == 1:
            X = X[np.newaxis, :]
        # Get candidate subsequences in X
        X_subs = get_all_subsequences(X, self.length, self.dilation)
        # Normalize candidates and shapelet values
        if self.normalize:
            X_means, X_stds = sliding_mean_std_one_series(X, self.length, self.dilation)
            X_subs = normalize_subsequences(X_subs, X_means, X_stds)
            _values = (self.values - self.values.mean(axis=-1)) / self.values.std(
                axis=1
            )
        else:
            _values = self.values

        # Compute distance vector
        c = compute_shapelet_dist_vector(X, _values, self.length)

        if ax is None:
            sns.set()
            sns.set_context(seaborn_context)
            fig = plt.figure(figsize=(figsize))
            plt.plot(c, label=label)
            if self.threshold is not None:
                plt.hlines(self.threshold, 0, c.shape[0], color=c_threshold)
            return fig
        else:
            ax.plot(c, label=label)
            if self.threshold is not None:
                ax.hlines(self.threshold, 0, c.shape[0], color=c_threshold)
            return ax


class ShapeletTransformerVisualizer:
    """
    A class to visualize the result from a fitted shapelet transformer.

    Parameters
    ----------
    estimator : object
        A fitted shapelet transformer.

    Returns
    -------
    None.

    """

    def __init__(self, estimator):
        self.estimator = estimator

    def _get_shapelet(self, id_shapelet):
        if isinstance(self.estimator, RandomDilatedShapeletTransform):
            length_ = self.estimator.shapelets_[1][id_shapelet]
            values_ = self.estimator.shapelets_[0][id_shapelet, :length_]
            dilation_ = self.estimator.shapelets_[2][id_shapelet]
            threshold_ = self.estimator.shapelets_[3][id_shapelet]
            normalize_ = self.estimator.shapelets_[4][id_shapelet]

        elif isinstance(self.estimator, RSAST):
            values_ = self.estimator._kernel_orig[id_shapelet][np.newaxis, :]
            length_ = values_.shape[1]
            dilation_ = 1
            normalize_ = True
            threshold_ = None

        elif isinstance(self.estimator, SAST):
            values_ = self.estimator._kernel_orig[id_shapelet][np.newaxis, :]
            length_ = values_.shape[1]
            dilation_ = 1
            normalize_ = True
            threshold_ = None

        elif isinstance(self.estimator, RandomShapeletTransform):
            values_ = self.estimator.shapelets[id_shapelet][6]
            length_ = self.estimator.shapelets[id_shapelet][1]
            dilation_ = 1
            normalize_ = True
            threshold_ = None
        else:
            raise NotImplementedError(
                "The provided estimator of class {} is not supported. Is it a shapelet"
                " transformer ?"
            )
        return _Shapelet(
            values_, length_, normalize_, dilation=dilation_, threshold=threshold_
        )

    def plot_on_X(
        self,
        id_shapelet,
        X,
        figsize=(10, 5),
        seaborn_context="talk",
        shp_dot_size=40,
        shp_c="purple",
        ax=None,
        label=None,
        x_linewidth=2.0,
    ):
        """
        Plot the shapelet on its best match on the time series X.

        Parameters
        ----------
        id_shapelet : int
            ID of the shapelet to plot.
        X : array, shape=(n_timestamps) or shape=(n_features, n_timestamps)
            Input time series
        figsize : tuple, optional
            Size of the figure. The default is (10,5).
        seaborn_context : str, optional
            Seaborn context. The default is 'talk'.
        alpha : float, optional
            Alpha parameter for plotting X. The default is 0.9.
        shp_dot_size : float, optional
            Size of the scatter plot to represent the shapelet on X.
            The default is 40.
        shp_c : str, optional
            Color of the shapelet scatter plot. The default is 'purple'.
        ax : matplotlib axe, optional
            A matplotlib axe on which to plot the figure. The default is None
            and will create a new figure of size figsize.
        label : str, optional
            Custom label to plot as legend for X. The default is None.
        x_linewidth : float, optional
            The linewidth of X plot. The default is 2.

        Returns
        -------
        fig : matplotlib figure
            The resulting figure with S on its best match on X. A normalized
            shapelet will be scalled to macth the scale of X.

        """
        return self._get_shapelet(id_shapelet).plot_on_X(
            X,
            figsize=figsize,
            seaborn_context=seaborn_context,
            shp_dot_size=shp_dot_size,
            shp_c=shp_c,
            ax=ax,
            label=label,
            x_linewidth=x_linewidth,
        )

    def plot_distance_vector(
        self,
        id_shapelet,
        X,
        figsize=(10, 5),
        seaborn_context="talk",
        c_threshold="purple",
        ax=None,
        label=None,
    ):
        """
        Plot the shapelet distance vector computed between itself and X.

        Parameters
        ----------
        id_shapelet : int
            ID of the shapelet to plot.
        X : array, shape=(n_timestamps) or shape=(n_features, n_timestamps)
            Input time series
        figsize : tuple, optional
            Size of the figure. The default is (10,5).
        seaborn_context : str, optional
            Seaborn context. The default is 'talk'.
        c_threshold : float, optional
            Color used to represent a line on the y-axis to visualize the lambda
            threshold. The default is 'purple'.
        ax : matplotlib axe, optional
            A matplotlib axe on which to plot the figure. The default is None
            and will create a new figure of size figsize.
        label : str, optional
            Custom label to plot as legend. The default is None.

        Returns
        -------
        fig : matplotlib figure
            The resulting figure with the distance vector obtained by d(S,X)

        """
        return self._get_shapelet(id_shapelet).plot_distance_vector(
            X,
            figsize=figsize,
            seaborn_context=seaborn_context,
            c_threshold=c_threshold,
            ax=ax,
            label=label,
        )

    def plot(self, id_shapelet, figsize=(10, 5), seaborn_context="talk", ax=None):
        """
        Plot the shapelet values.

        Parameters
        ----------
        id_shapelet : int
            ID of the shapelet to plot.
        figsize : tuple, optional
            2D size of the figure. The default is (10,5).
        seaborn_context : str, optional
            Seaborn module context. The default is 'talk'.
        ax : matplotlib axe, optional
            A matplotlib axe on which to plot the figure. The default is None
            and will create a new figure of size figsize.

        Returns
        -------
        fig : matplotlib figure
            The resulting figure
        """
        return self._get_shapelet(id_shapelet).plot(
            figsize=figsize, seaborn_context=seaborn_context, ax=ax
        )


class ShapeletClassifierVisualizer:
    """
    A class to visualize the result from a fitted shapelet classifier.

    Parameters
    ----------
    estimator : object
        A fitted shapelet classifier.
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self.transformer_vis = ShapeletTransformerVisualizer(
            self.estimator._transformer
        )

    def _get_shp_importance(self, class_id):
        if isinstance(self.estimator, RDSTClassifier):
            coefs = self.estimator._estimator["ridgeclassifiercv"].coef_
            n_classes = coefs.shape[0]
            if n_classes == 1:
                coefs = np.append(-coefs, coefs, axis=0)
            c_ = np.zeros(self.RDST_Ridge.transformer.shapelets_[1].shape[0] * 3)
            c_[self.estimator._estimator["c_standardscaler"].usefull_atts] = coefs[
                class_id
            ]
            return c_
        elif isinstance(self.estimator, RSASTClassifier):
            raise NotImplementedError()
        elif isinstance(self.estimator, SASTClassifier):
            raise NotImplementedError()
        elif isinstance(self.estimator, ShapeletTransformClassifier):
            raise NotImplementedError()
        elif isinstance(self.estimator, LearningShapeletClassifier):
            raise NotImplementedError()
        else:
            raise NotImplementedError(
                "The provided estimator of class {} is not supported. Is it a shapelet"
                " transformer ?"
            )
        return c_

    def visualize_best_shapelets_one_class(
        self, X, y, class_id, n_shp=1, figsize=(16, 12), seaborn_context="talk"
    ):
        """
        Plot the n_shp best candidates for the class_id.

        Visualize best macth on two random samples and how the shapelet discriminate
        (X,y) with boxplots.

        Parameters
        ----------
        X : array, shape=(n_samples, n_fetaures, n_timestamps)
            A time series dataset. Can be the training set to visualize training
            results, or testing to visualize generalization to unseen samples.
        y : array, shape=(n_samples)
            The true classes of the time series dataset.
        class_id : int
            ID of the class we want to visualize. The n_shp best shapelet for
            this class will be selected based on the feature coefficients
            inside the ridge classifier.
        n_shp : int, optional
            Number of plots to output, one per shapelet (i.e. the n_shp best shapelets
            for class_id). The default is 1.
        figsize : tuple, optional
            Size of the figure. The default is (16,12).
        seaborn_context : str, optional
            Seaborn context. The default is 'talk'.

        Returns
        -------
        None.

        """
        # TODO : adapt this method to all shapelet classifiers
        sns.set()
        sns.set_context(seaborn_context)

        coefs = self._get_shp_importance(class_id)
        idx = (coefs.argsort() // 3)[::-1]

        shp_ids = []
        i = 0
        while len(shp_ids) < n_shp and i < idx.shape[0]:
            if idx[i] not in shp_ids:
                shp_ids = shp_ids + [idx[i]]
            i += 1

        X_new = self.estimator._transformer.transform(X)
        i_example = np.random.choice(np.where(y == class_id)[0])
        i_example2 = np.random.choice(np.where(y != class_id)[0])
        y_copy = (y == class_id).astype(int)
        for i_shp in shp_ids:
            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=figsize)

            ax[0, 0].set_title("Boxplot of min")
            sns.boxplot(x=y_copy, y=X_new[:, (i_shp * 3)], ax=ax[0, 0])
            ax[0, 0].set_xticklabels(["Other classes", f"Class {class_id}"])

            ax[0, 1].set_title("Boxplot of argmin")
            sns.boxplot(x=y_copy, y=X_new[:, 1 + (i_shp * 3)], ax=ax[0, 1])

            ax[0, 1].set_xticklabels(["Other classes", f"Class {class_id}"])
            ax[0, 2].set_title("Boxplot of Shapelet Occurence")
            sns.boxplot(x=y_copy, y=X_new[:, 2 + (i_shp * 3)], ax=ax[0, 2])

            ax[0, 2].set_xticklabels(["Other classes", f"Class {class_id}"])

            ax[1, 0].set_title("Best match")
            ax[1, 2].set_title("Distance vectors")

            self.transformer_vis.plot(i_shp, ax=ax[1, 1])

            self.transformer_vis.plot_on_X(
                i_shp, X[i_example2], ax=ax[1, 0], label="Other class"
            )
            self.transformer_vis.plot_on_X(
                i_shp, X[i_example], ax=ax[1, 0], label=f"Class {class_id}"
            )

            self.transformer_vis.plot_distance_vector(
                i_shp, X[i_example2], ax=ax[1, 2], label="Other class"
            )
            self.transformer_vis.plot_distance_vector(
                i_shp, X[i_example], ax=ax[1, 2], label=f"Class {class_id}"
            )

            ax[1, 0].legend()
            plt.show()

    def plot_on_X(
        self,
        id_shapelet,
        X,
        figsize=(10, 5),
        seaborn_context="talk",
        shp_dot_size=40,
        shp_c="purple",
        ax=None,
        label=None,
        x_linewidth=2.0,
    ):
        """
        Plot the shapelet on its best match on the time series X.

        Parameters
        ----------
        id_shapelet : int
            ID of the shapelet to plot.
        X : array, shape=(n_timestamps) or shape=(n_features, n_timestamps)
            Input time series
        figsize : tuple, optional
            Size of the figure. The default is (10,5).
        seaborn_context : str, optional
            Seaborn context. The default is 'talk'.
        alpha : float, optional
            Alpha parameter for plotting X. The default is 0.9.
        shp_dot_size : float, optional
            Size of the scatter plot to represent the shapelet on X.
            The default is 40.
        shp_c : str, optional
            Color of the shapelet scatter plot. The default is 'purple'.
        ax : matplotlib axe, optional
            A matplotlib axe on which to plot the figure. The default is None
            and will create a new figure of size figsize.
        label : str, optional
            Custom label to plot as legend for X. The default is None.
        x_linewidth : float, optional
            The linewidth of X plot. The default is 2.

        Returns
        -------
        fig : matplotlib figure
            The resulting figure with S on its best match on X. A normalized
            shapelet will be scalled to macth the scale of X.

        """
        self.transformer_vis.plot_on_X(
            id_shapelet,
            X,
            figsize=figsize,
            seaborn_context=seaborn_context,
            shp_dot_size=shp_dot_size,
            shp_c=shp_c,
            ax=ax,
            label=label,
            x_linewidth=x_linewidth,
        )

    def plot_distance_vector(
        self,
        id_shapelet,
        X,
        figsize=(10, 5),
        seaborn_context="talk",
        c_threshold="purple",
        ax=None,
        label=None,
    ):
        """
        Plot the shapelet distance vector computed between itself and X.

        Parameters
        ----------
        id_shapelet : int
            ID of the shapelet to plot.
        X : array, shape=(n_timestamps) or shape=(n_features, n_timestamps)
            Input time series
        figsize : tuple, optional
            Size of the figure. The default is (10,5).
        seaborn_context : str, optional
            Seaborn context. The default is 'talk'.
        c_threshold : float, optional
            Color used to represent a line on the y-axis to visualize the lambda
            threshold. The default is 'purple'.
        ax : matplotlib axe, optional
            A matplotlib axe on which to plot the figure. The default is None
            and will create a new figure of size figsize.
        label : str, optional
            Custom label to plot as legend. The default is None.

        Returns
        -------
        fig : matplotlib figure
            The resulting figure with the distance vector obtained by d(S,X)

        """
        self.transformer_vis.plot_distance_vector(
            id_shapelet,
            X,
            figsize=figsize,
            seaborn_context=seaborn_context,
            c_threshold=c_threshold,
            ax=ax,
            label=label,
        )

    def plot(self, id_shapelet, figsize=(10, 5), seaborn_context="talk", ax=None):
        """
        Plot the shapelet values.

        Parameters
        ----------
        id_shapelet : int
            ID of the shapelet to plot.
        figsize : tuple, optional
            2D size of the figure. The default is (10,5).
        seaborn_context : str, optional
            Seaborn module context. The default is 'talk'.
        ax : matplotlib axe, optional
            A matplotlib axe on which to plot the figure. The default is None
            and will create a new figure of size figsize.

        Returns
        -------
        fig : matplotlib figure
            The resulting figure
        """
        self.transformer_vis.plot_distance_vector(
            id_shapelet, figsize=figsize, seaborn_context=seaborn_context, ax=ax
        )
