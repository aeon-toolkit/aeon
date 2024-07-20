"""Shapelet plotting tools."""

__maintainer__ = ["baraline"]

__all__ = ["ShapeletClassifierVisualizer", "ShapeletTransformerVisualizer"]

import copy

import numpy as np
from sklearn.ensemble._forest import BaseForest
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.tree import BaseDecisionTree

from aeon.classification.shapelet_based import (
    RDSTClassifier,
    RSASTClassifier,
    SASTClassifier,
    ShapeletTransformClassifier,
)
from aeon.distances import get_distance_function
from aeon.transformations.collection.shapelet_based import (
    RSAST,
    SAST,
    RandomDilatedShapeletTransform,
    RandomShapeletTransform,
)
from aeon.transformations.collection.shapelet_based._dilated_shapelet_transform import (
    compute_shapelet_dist_vector,
    get_all_subsequences,
    normalize_subsequences,
)
from aeon.utils.numba.general import sliding_mean_std_one_series
from aeon.utils.validation._dependencies import _check_soft_dependencies


class ShapeletVisualizer:
    """
    A Shapelet object to use for ploting operations.

    Parameters
    ----------
    values : array, shape=(n_channels, length)
        Values of the shapelet.
    normalize : bool
        Wheter the shapelet use a normalized distance.
    dilation : int
        Dilation of the shapelet. The default is 1, which is equivalent to no
        dilation.
    threshold : float
        Lambda threshold for Shapelet Occurrence feature. The default value is None
        if it is not used (used in RDST).
    length : int
        Length of the shapelet. The default values is None, meaning length is infered
        from the values array. Otherwise, the values array 2nd axis will be set to this
        length.

    """

    def __init__(
        self,
        values,
        normalize=False,
        dilation=1,
        threshold=None,
        length=None,
        distance="euclidean",
    ):
        self.values = np.asarray(values)
        if self.values.ndim == 1:
            self.values = self.values[np.newaxis, :]
        if length is None:
            self.length = self.values.shape[1]
        else:
            self.values = self.values[:, :length]
            self.length = length
        self.n_channels = self.values.shape[0]
        self.normalize = normalize
        self.threshold = threshold
        self.dilation = dilation
        self.distance_func = get_distance_function(distance)

    def plot(
        self,
        ax=None,
        scatter_options={  # noqa: B006
            "s": 70,
            "alpha": 0.75,
            "zorder": 3,
            "edgecolor": "black",
            "linewidths": 2,
        },
        plot_options={  # noqa: B006
            "linewidth": 2,
            "alpha": 0.9,
            "linestyle": "--",
        },
        figure_options={  # noqa: B006
            "figsize": (10, 5),
            "dpi": 100,
        },
        rc_Params_options={"font.size": 22},  # noqa: B006
        matplotlib_style="seaborn-v0_8",
        custom_title_string=None,
    ):
        """
        Plot the shapelet values.

        Parameters
        ----------
        ax : matplotlib axe
            A matplotlib axe on which to plot the figure. The default is None
            and will create a new figure of size figsize.
        plot_options : dict
            Options to apply to plot of the shapelet values.
        scatter_options : dict
            Options to apply to scatter plot of the shapelet values.
        figure_options : dict
            Dictionnary of options passed to plt.figure. Only used if ax is None.
        rc_Params_options: dict
            Dictionnary of options passed to plt.rcParams.update. Only used if ax is
            None.
        matplotlib_style: str
            Matplotlib style to be used. Only used if ax is None.
        custom_title_string : str
            If not None, use this string as title for the plot instead of the default
            one based on the shapelet parametres.

        Returns
        -------
        fig : matplotlib figure
            The resulting figure
        """
        _check_soft_dependencies("matplotlib")
        import matplotlib.pyplot as plt

        if "label" not in plot_options.keys():
            plot_options["label"] = ""
        if custom_title_string is None:
            title_string = "Shapelet params:"
            if self.dilation > 1:
                title_string += f" dilation={self.dilation}"
            if self.threshold is not None:
                title_string += f" threshold={np.round(self.threshold, decimals=2)}"
            title_string += f" normalize={self.normalize}"
        else:
            title_string = custom_title_string
        if ax is None:
            plt.style.use(matplotlib_style)
            plt.rcParams.update(rc_Params_options)

            fig = plt.figure(**figure_options)
            for i in range(self.n_channels):
                if self.n_channels > 1:
                    plot_options.update(
                        {"label": str(plot_options["label"]) + f" channel {i}"}
                    )
                plt.plot(self.values[i], **plot_options)
                plt.scatter(np.arange(self.length), self.values[i], **scatter_options)
            plt.ylabel("shapelet values")
            plt.xlabel("timepoint")
            plt.title(title_string)
            plt.legend()
            return fig
        else:
            for i in range(self.n_channels):
                if self.n_channels > 1:
                    plot_options.update(
                        {"label": str(plot_options["label"]) + f" channel {i}"}
                    )
                ax.plot(self.values[i], **plot_options)
                ax.scatter(np.arange(self.length), self.values[i], **scatter_options)
            ax.set_title(title_string)
            ax.set_ylabel("shapelet values")
            ax.set_xlabel("timepoint")
            ax.legend()
            return ax

    def plot_on_X(
        self,
        X,
        ax=None,
        shp_scatter_options={  # noqa: B006
            "s": 40,
            "c": "purple",
            "alpha": 0.9,
            "zorder": 3,
        },
        x_plot_options={"linewidth": 2, "alpha": 0.9},  # noqa: B006
        figure_options={  # noqa: B006
            "figsize": (10, 5),
            "dpi": 100,
        },
        rc_Params_options={"font.size": 22},  # noqa: B006
        matplotlib_style="seaborn-v0_8",
    ):
        """
        Plot the shapelet on its best match on the time series X.

        Parameters
        ----------
        X : array, shape=(n_features, n_timestamps)
            Input time series
        ax : matplotlib axe
            A matplotlib axe on which to plot the figure. The default is None
            and will create a new figure of size figsize.
        shp_scatter_options : dict
            Dictionnary of options passed to the scatter plot of the shapelet values.
        x_plot_options : dict
            Dictionnary of options passed to the plot of the time series values.
        figure_options : dict
            Dictionnary of options passed to plt.figure. Only used if ax is None.
        rc_Params_options: dict
            Dictionnary of options passed to plt.rcParams.update. Only used if ax is
            None.
        matplotlib_style: str
            Matplotlib style to be used. Only used if ax is None.

        Returns
        -------
        fig : matplotlib figure
            The resulting figure with S on its best match on X. A normalized
            shapelet will be scalled to macth the scale of X.

        """
        _check_soft_dependencies("matplotlib")
        import matplotlib.pyplot as plt

        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        if "label" not in x_plot_options.keys():
            x_plot_options["label"] = ""

        # Get candidate subsequences in X
        X_subs = get_all_subsequences(X, self.length, self.dilation)

        # Normalize candidates and shapelet values
        if self.normalize:
            X_means, X_stds = sliding_mean_std_one_series(X, self.length, self.dilation)
            X_subs = normalize_subsequences(X_subs, X_means, X_stds)
            _values = (
                self.values - self.values.mean(axis=-1, keepdims=True)
            ) / self.values.std(axis=-1, keepdims=True)
        else:
            _values = self.values

        # Compute distance vector
        c = compute_shapelet_dist_vector(
            X_subs, _values, self.length, self.distance_func
        )

        # Get best match index
        idx_best = c.argmin()
        idx_match = np.asarray(
            [(idx_best + i * self.dilation) % X.shape[1] for i in range(self.length)]
        ).astype(int)

        # If normalize, scale back the values of the shapelet to the scale of the match
        if self.normalize:
            _values = (_values * X[:, idx_match].std(axis=-1, keepdims=True)) + X[
                :, idx_match
            ].mean(axis=-1, keepdims=True)

        if ax is None:
            plt.style.use(matplotlib_style)
            plt.rcParams.update(rc_Params_options)

            fig = plt.figure(**figure_options)
            for i in range(self.n_channels):
                if self.n_channels > 1:
                    x_plot_options.update(
                        {"label": str(x_plot_options["label"]) + f" channel {i}"}
                    )
                plt.plot(X[i], **x_plot_options)
                plt.scatter(idx_match, _values[i], **shp_scatter_options)
                plt.title("Best match of shapelet on X")
            plt.ylabel("shapelet values")
            plt.xlabel("timepoint")
            return fig
        else:
            for i in range(self.n_channels):
                if self.n_channels > 1:
                    x_plot_options.update(
                        {"label": str(x_plot_options["label"]) + f" channel {i}"}
                    )
                ax.plot(X[i], **x_plot_options)
                ax.scatter(idx_match, _values[i], **shp_scatter_options)
            ax.set_ylabel("shapelet values")
            ax.set_xlabel("timepoint")
            return ax

    def plot_distance_vector(
        self,
        X,
        ax=None,
        show_legend=True,
        show_threshold=True,
        dist_plot_options={"linewidth": 2, "alpha": 0.9},  # noqa: B006
        threshold_plot_options={  # noqa: B006
            "linewidth": 2,
            "alpha": 0.9,
            "color": "purple",
            "label": "threshold",
        },
        figure_options={  # noqa: B006
            "figsize": (10, 5),
            "dpi": 100,
        },
        rc_Params_options={"font.size": 22},  # noqa: B006
        matplotlib_style="seaborn-v0_8",
    ):
        """
        Plot the shapelet distance vector computed between itself and X.

        Parameters
        ----------
        X : array, shape=(n_features, n_timestamps)
            Input time series
        ax : matplotlib axe
            A matplotlib axe on which to plot the figure. The default is None
            and will create a new figure of size figsize.
        show_legend : bool, optional
            Wheter to show legend. Default is True
        show_threshold: bool, optional
            Wheter to show threshold (if it is not set to None). Default is True.
        threshold_plot_options : dict
            Dictionnary of options passed to the line plot of the threshold.
        dist_plot_options : dict
            Dictionnary of options passed to the plot of the distance vector values.
        figure_options : dict
            Dictionnary of options passed to plt.figure. Only used if ax is None.
        rc_Params_options: dict
            Dictionnary of options passed to plt.rcParams.update. Only used if ax is
            None.
        matplotlib_style: str
            Matplotlib style to be used. Only used if ax is None.

        Returns
        -------
        fig : matplotlib figure
            The resulting figure with the distance vector obtained by d(S,X)

        """
        _check_soft_dependencies("matplotlib")
        import matplotlib.pyplot as plt

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

        c = compute_shapelet_dist_vector(
            X_subs, _values, self.length, self.distance_func
        )

        if ax is None:
            plt.style.use(matplotlib_style)
            plt.rcParams.update(rc_Params_options)
            fig = plt.figure(**figure_options)

            plt.plot(c, **dist_plot_options)
            if self.threshold is not None and show_threshold:
                plt.hlines(self.threshold, 0, c.shape[0], **threshold_plot_options)
            plt.title("Distance vector between shapelet and X")
            if show_legend:
                plt.legend()
            return fig
        else:
            ax.plot(c, **dist_plot_options)
            if self.threshold is not None and show_threshold:
                ax.hlines(self.threshold, 0, c.shape[0], **threshold_plot_options)
            return ax


class ShapeletTransformerVisualizer:
    """
    A class to visualize the result from a fitted shapelet transformer.

    Parameters
    ----------
    estimator : object
        A fitted shapelet transformer.

    """

    def __init__(self, estimator):
        self.estimator = estimator

    def _get_shapelet(self, id_shapelet):
        if isinstance(self.estimator, RandomDilatedShapeletTransform):
            length_ = self.estimator.shapelets_[1][id_shapelet]
            values_ = self.estimator.shapelets_[0][id_shapelet]
            dilation_ = self.estimator.shapelets_[2][id_shapelet]
            threshold_ = self.estimator.shapelets_[3][id_shapelet]
            normalize_ = self.estimator.shapelets_[4][id_shapelet]
            distance = self.estimator.distance

        elif isinstance(self.estimator, (RSAST, SAST)):
            values_ = self.estimator._kernel_orig[id_shapelet]
            length_ = values_.shape[0]
            dilation_ = 1
            normalize_ = True
            threshold_ = None
            distance = "euclidean"

        elif isinstance(self.estimator, RandomShapeletTransform):
            values_ = self.estimator.shapelets[id_shapelet][6]
            length_ = self.estimator.shapelets[id_shapelet][1]
            dilation_ = 1
            normalize_ = True
            threshold_ = None
            distance = "euclidean"
        else:
            raise NotImplementedError(
                "The provided estimator of type {type(self.estimator)} is not supported"
                ". Is it a shapelet transformer ?"
            )
        return ShapeletVisualizer(
            values_,
            normalize=normalize_,
            dilation=dilation_,
            threshold=threshold_,
            length=length_,
            distance=distance,
        )

    def plot_on_X(
        self,
        id_shapelet,
        X,
        ax=None,
        shp_scatter_options={  # noqa: B006
            "s": 40,
            "c": "purple",
            "alpha": 0.9,
            "zorder": 3,
        },
        x_plot_options={"linewidth": 2, "alpha": 0.9},  # noqa: B006
        figure_options={  # noqa: B006
            "figsize": (10, 5),
            "dpi": 100,
        },
        rc_Params_options={"font.size": 22},  # noqa: B006
        matplotlib_style="seaborn-v0_8",
    ):
        """
        Plot the shapelet on its best match on the time series X.

        Parameters
        ----------
        id_shapelet : int
            ID of the shapelet to plot.
        X : array, shape=(n_features, n_timestamps)
            Input time series
        ax : matplotlib axe
            A matplotlib axe on which to plot the figure. The default is None
            and will create a new figure of size figsize.
        shp_scatter_options : dict
            Dictionnary of options passed to the scatter plot of the shapelet values.
        x_plot_options : dict
            Dictionnary of options passed to the plot of the time series values.
        figure_options : dict
            Dictionnary of options passed to plt.figure. Only used if ax is None.
        rc_Params_options: dict
            Dictionnary of options passed to plt.rcParams.update. Only used if ax is
            None.
        matplotlib_style: str
            Matplotlib style to be used. Only used if ax is None.

        Returns
        -------
        fig : matplotlib figure
            The resulting figure with S on its best match on X. A normalized
            shapelet will be scalled to macth the scale of X.

        """
        return self._get_shapelet(id_shapelet).plot_on_X(
            X,
            ax=ax,
            shp_scatter_options=shp_scatter_options,
            x_plot_options=x_plot_options,
            figure_options=figure_options,
            rc_Params_options=rc_Params_options,
            matplotlib_style=matplotlib_style,
        )

    def plot_distance_vector(
        self,
        id_shapelet,
        X,
        ax=None,
        show_legend=True,
        show_threshold=True,
        dist_plot_options={"linewidth": 2, "alpha": 0.9},  # noqa: B006
        threshold_plot_options={  # noqa: B006
            "linewidth": 2,
            "alpha": 0.9,
            "color": "purple",
            "label": "threshold",
        },
        figure_options={  # noqa: B006
            "figsize": (10, 5),
            "dpi": 100,
        },
        rc_Params_options={"font.size": 22},  # noqa: B006
        matplotlib_style="seaborn-v0_8",
    ):
        """
        Plot the shapelet distance vector computed between itself and X.

        Parameters
        ----------
        id_shapelet : int
            ID of the shapelet to plot.
        X : array, shape=(n_timestamps) or shape=(n_features, n_timestamps)
            Input time series
        ax : matplotlib axe
            A matplotlib axe on which to plot the figure. The default is None
            and will create a new figure of size figsize.
        show_legend : bool, optional
            Wheter to show legend. Default is True
        show_threshold: bool, optional
            Wheter to show threshold (if it is not set to None). Default is True.
        threshold_plot_options : dict
            Dictionnary of options passed to the line plot of the threshold.
        dist_plot_options : dict
            Dictionnary of options passed to the plot of the distance vector values.
        figure_options : dict
            Dictionnary of options passed to plt.figure. Only used if ax is None.
        rc_Params_options: dict
            Dictionnary of options passed to plt.rcParams.update. Only used if ax is
            None.
        matplotlib_style: str
            Matplotlib style to be used. Only used if ax is None.

        Returns
        -------
        fig : matplotlib figure
            The resulting figure with the distance vector obtained by d(S,X)

        """
        return self._get_shapelet(id_shapelet).plot_distance_vector(
            X,
            ax=ax,
            show_legend=show_legend,
            show_threshold=show_threshold,
            threshold_plot_options=threshold_plot_options,
            dist_plot_options=dist_plot_options,
            figure_options=figure_options,
            rc_Params_options=rc_Params_options,
            matplotlib_style=matplotlib_style,
        )

    def plot(
        self,
        id_shapelet,
        ax=None,
        scatter_options={  # noqa: B006
            "s": 70,
            "alpha": 0.75,
            "zorder": 3,
            "edgecolor": "black",
            "linewidths": 2,
        },
        plot_options={  # noqa: B006
            "linewidth": 2,
            "alpha": 0.9,
            "linestyle": "--",
        },
        figure_options={  # noqa: B006
            "figsize": (10, 5),
            "dpi": 100,
        },
        rc_Params_options={"font.size": 22},  # noqa: B006
        matplotlib_style="seaborn-v0_8",
        custom_title_string=None,
    ):
        """
        Plot the shapelet values.

        Parameters
        ----------
        id_shapelet : int
            ID of the shapelet to plot.
        ax : matplotlib axe
            A matplotlib axe on which to plot the figure. The default is None
            and will create a new figure of size figsize.
         scatter_options : dict
             Options to apply to scatter plot of the shapelet values.
         figure_options : dict
             Dictionnary of options passed to plt.figure. Only used if ax is None.
        figure_options : dict
            Dictionnary of options passed to plt.figure. Only used if ax is None.
        rc_Params_options: dict
            Dictionnary of options passed to plt.rcParams.update. Only used if ax is
            None.
        matplotlib_style: str
            Matplotlib style to be used. Only used if ax is None.
        custom_title_string : str
            If not None, use this string as title for the plot instead of the default
            one based on the shapelet parametres.

        Returns
        -------
        fig : matplotlib figure
            The resulting figure
        """
        return self._get_shapelet(id_shapelet).plot(
            ax=ax,
            plot_options=plot_options,
            scatter_options=scatter_options,
            figure_options=figure_options,
            rc_Params_options=rc_Params_options,
            matplotlib_style=matplotlib_style,
            custom_title_string=custom_title_string,
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
        """
        Return the shapelet importance for a speficied class.

        Parameters
        ----------
        class_id : int
            Class identifier used. If the computation method used for importance is
            independent of the class, this will be ignored.

        Raises
        ------
        NotImplementedError
            Raise this error when an estimator given in init is not supported.

        Returns
        -------
        idx : array, shape = (n_shapelets)
            Sorted shapelet index from best to worse
        coefs : array, shape = (n_shapelets)
            Importance values linked to each shapelets.

        """
        if isinstance(self.estimator, (RDSTClassifier, ShapeletTransformClassifier)):
            classifier = self.estimator._estimator
        elif isinstance(self.estimator, (RSASTClassifier, SASTClassifier)):
            classifier = self.estimator._classifier
        else:
            raise NotImplementedError(
                f"The provided estimator of type {type(self.estimator)} is not"
                " supported. Is it a shapelet classifier ?"
            )
        # If classifier is a pipeline, get the last step (i.e. the classifier)
        if isinstance(classifier, Pipeline):
            classifier = classifier[-1]

        # This suppose that the higher the coef linked to each feature, the most
        # impact this feature makes on classification for the given class_id
        if isinstance(classifier, LinearClassifierMixin):
            coefs = classifier.coef_
            n_classes = coefs.shape[0]
            if n_classes == 1:
                coefs = np.append(-coefs, coefs, axis=0)
            coefs = coefs[class_id]
            idx = coefs.argsort()[::-1]

        elif isinstance(classifier, (BaseForest, BaseDecisionTree)):
            coefs = classifier.feature_importances_
            idx = coefs.argsort()[::-1]
        else:
            raise NotImplementedError(
                f"The classifier linked to the estimator is not supported. We expect a "
                "classifier inheriting from LinearClassifierMixin, BaseForest or "
                f"BaseDecisionTree but got {type(classifier)}"
            )

        coefs = coefs[idx]
        if isinstance(self.estimator, RDSTClassifier):
            # As each shapelet generate 3 features, divide feature id by 3 so all
            # features generated by one shapelet share the same ID
            idx = idx // 3
        return idx, coefs

    def _get_boxplot_data(self, X, mask_class_id, mask_other_class_id, id_shp):
        if isinstance(self.estimator, RDSTClassifier):
            titles = [
                "Boxplot of min",
                "Boxplot of argmin",
                "Boxplot of Shapelet Occurence",
            ]
            for i in range(3):
                box_data = [
                    X[mask_other_class_id, i + (id_shp * 3)],
                    X[mask_class_id, i + (id_shp * 3)],
                ]
                yield titles[i], box_data

        elif isinstance(
            self.estimator,
            (RSASTClassifier, SASTClassifier, ShapeletTransformClassifier),
        ):
            titles = [
                "Boxplot of min",
            ]
            box_data = [
                X[mask_other_class_id, id_shp],
                X[mask_class_id, id_shp],
            ]
            yield titles[0], box_data

        else:
            raise NotImplementedError(
                f"The provided estimator of type {type(self.estimator)} is not"
                " supported. Is it a shapelet classifier ?"
            )

    def visualize_best_shapelets_one_class(
        self,
        X,
        y,
        class_id,
        n_shp=1,
        id_example_other=None,
        id_example_class=None,
        class_colors=("tab:green", "tab:orange"),
        shp_scatter_options={  # noqa: B006
            "s": 70,
            "alpha": 0.75,
            "zorder": 1,
            "edgecolor": "black",
            "linewidths": 2,
        },
        x_plot_options={"linewidth": 2, "alpha": 0.9},  # noqa: B006
        shp_plot_options={  # noqa: B006
            "linewidth": 2,
            "alpha": 0.9,
            "linestyle": "--",
        },
        dist_plot_options={"linewidth": 2, "alpha": 0.9},  # noqa: B006
        threshold_plot_options={  # noqa: B006
            "linewidth": 2,
            "alpha": 0.9,
            "color": "purple",
            "label": "threshold",
        },
        boxplot_options={  # noqa: B006
            "patch_artist": True,
            "widths": 0.6,
            "showmeans": True,
            "meanline": True,
            "boxprops": {"linewidth": 1.5},
            "whiskerprops": {"linewidth": 1.5},
            "medianprops": {"linewidth": 1.5, "color": "black"},
            "meanprops": {"linewidth": 1.5, "color": "black"},
            "flierprops": {"linewidth": 1.5},
        },
        figure_options={  # noqa: B006
            "figsize": (20, 12),
            "nrows": 2,
            "ncols": 3,
            "dpi": 200,
        },
        rc_Params_options={  # noqa: B006
            "legend.fontsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
        },
        matplotlib_style="seaborn-v0_8",
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
            inside the ridge classifier. The original labels are given to a
            LabelEncoder, hence why we ask for an integer ID.
        n_shp : int, optional
            Number of plots to output, one per shapelet (i.e. the n_shp best shapelets
            for class_id). The default is 1.
        id_example_other : int
            Sample ID to use for sample of other class. If None, a random one is
            selected.
        id_example_class : int
            Sample ID to use for sample of class_id. If None, a random one is selected.
        shp_scatter_options : dict
            Dictionnary of options passed to the scatter plot of the shapelet values.
        x_plot_options : dict
            Dictionnary of options passed to the plot of the time series values.
        shp_plot_options : dict
            Dictionnary of options passed to the plot of the shapelet values.
        threshold_plot_options : dict
            Dictionnary of options passed to the line plot of the threshold.
        dist_plot_options : dict
            Dictionnary of options passed to the plot of the distance vector values.
        figure_options : dict
            Dictionnary of options passed to plt.figure.
        boxplot_options : dict
            Dictionnary of options passed to features boxplot.
        rc_Params_options: dict
            Dictionnary of options passed to plt.rcParams.update.
        matplotlib_style: str
            Matplotlib style to be used.

        Returns
        -------
        figures : list of matplotlib figure
            The resulting figures for each selected shapelets (list of size n_shp)
        """
        from sklearn.preprocessing import LabelEncoder

        _check_soft_dependencies("matplotlib")
        import matplotlib.pyplot as plt

        y = LabelEncoder().fit_transform(y)

        plt.style.use(matplotlib_style)
        plt.rcParams.update(**rc_Params_options)

        idx, _ = self._get_shp_importance(class_id)

        shp_ids = []
        i = 0
        while len(shp_ids) < n_shp and i < idx.shape[0]:
            if idx[i] not in shp_ids:
                shp_ids = shp_ids + [idx[i]]
            i += 1

        X_new = self.estimator._transformer.transform(X)
        mask_class_id = np.where(y == class_id)[0]
        mask_other_class_id = np.where(y != class_id)[0]
        if id_example_class is None:
            id_example_class = np.random.choice(mask_class_id)
        if id_example_other is None:
            id_example_other = np.random.choice(mask_other_class_id)
        figures = []
        for i_shp in shp_ids:
            fig, ax = plt.subplots(**figure_options)
            if ax.ndim == 1:
                n_cols = ax.shape[0]
            else:
                n_cols = ax.shape[1]

            # Plots of features boxplots
            i_ax = 0
            for title, box_data in self._get_boxplot_data(
                X_new, mask_class_id, mask_other_class_id, i_shp
            ):
                if ax.ndim == 1:
                    current_ax = ax[i_ax % n_cols]
                else:
                    current_ax = ax[i_ax // n_cols, i_ax % n_cols]
                current_ax.set_title(title)
                bplot = current_ax.boxplot(box_data, **boxplot_options)
                current_ax.set_xticklabels(["Other classes", f"Class {class_id}"])
                for patch, color in zip(bplot["boxes"], class_colors):
                    patch.set_facecolor(color)
                i_ax += 1

            # Plots of shapelet on X
            x0_plot_options = copy.deepcopy(x_plot_options)
            x0_plot_options.update(
                {
                    "label": f"Sample of class {y[id_example_other]}",
                    "c": class_colors[0],
                }
            )
            if ax.ndim == 1:
                current_ax = ax[i_ax % n_cols]
            else:
                current_ax = ax[i_ax // n_cols, i_ax % n_cols]
            shp0_scatter_options = copy.deepcopy(shp_scatter_options)
            shp0_scatter_options.update({"c": class_colors[0]})
            self.plot_on_X(
                i_shp,
                X[id_example_other],
                ax=current_ax,
                x_plot_options=x0_plot_options,
                shp_scatter_options=shp0_scatter_options,
            )

            x1_plot_options = copy.deepcopy(x_plot_options)
            x1_plot_options.update(
                {
                    "label": f"Sample of class {y[id_example_class]}",
                    "c": class_colors[1],
                }
            )
            shp1_scatter_options = copy.deepcopy(shp_scatter_options)
            shp1_scatter_options.update({"c": class_colors[1]})
            self.plot_on_X(
                i_shp,
                X[id_example_class],
                ax=current_ax,
                x_plot_options=x1_plot_options,
                shp_scatter_options=shp1_scatter_options,
            )
            current_ax.set_title("Best match on examples")
            current_ax.legend()

            # Plots of shapelet values
            i_ax += 1
            if ax.ndim == 1:
                current_ax = ax[i_ax % n_cols]
            else:
                current_ax = ax[i_ax // n_cols, i_ax % n_cols]
            self.plot(
                i_shp,
                ax=current_ax,
                plot_options=shp_plot_options,
                scatter_options=shp_scatter_options,
            )

            # Plots of distance vectors
            i_ax += 1
            if ax.ndim == 1:
                current_ax = ax[i_ax % n_cols]
            else:
                current_ax = ax[i_ax // n_cols, i_ax % n_cols]
            d0_plot_options = copy.deepcopy(dist_plot_options)
            d0_plot_options.update(
                {
                    "c": class_colors[0],
                    "label": f"Distance vector of class {y[id_example_other]}",
                }
            )
            self.plot_distance_vector(
                i_shp,
                X[id_example_other],
                ax=current_ax,
                show_legend=False,
                show_threshold=False,
                dist_plot_options=d0_plot_options,
            )
            d1_plot_options = copy.deepcopy(dist_plot_options)
            d1_plot_options.update(
                {
                    "c": class_colors[1],
                    "label": f"Distance vector of class {y[id_example_class]}",
                }
            )
            self.plot_distance_vector(
                i_shp,
                X[id_example_class],
                ax=current_ax,
                dist_plot_options=d1_plot_options,
            )
            current_ax.legend()
            current_ax.set_title("Distance vectors of examples")
            figures.append(fig)
        return figures

    def plot_on_X(
        self,
        id_shapelet,
        X,
        ax=None,
        shp_scatter_options={  # noqa: B006
            "s": 40,
            "c": "purple",
            "alpha": 0.9,
            "zorder": 3,
        },
        x_plot_options={"linewidth": 2, "alpha": 0.9},  # noqa: B006
        figure_options={  # noqa: B006
            "figsize": (10, 5),
            "dpi": 100,
        },
        rc_Params_options={"font.size": 22},  # noqa: B006
        matplotlib_style="seaborn-v0_8",
    ):
        """
        Plot the shapelet on its best match on the time series X.

        Parameters
        ----------
        id_shapelet : int
            ID of the shapelet to plot.
        X : array, shape=(n_features, n_timestamps)
            Input time series
        ax : matplotlib axe
            A matplotlib axe on which to plot the figure. The default is None
            and will create a new figure of size figsize.
        shp_scatter_options : dict
            Dictionnary of options passed to the scatter plot of the shapelet values.
        x_plot_options : dict
            Dictionnary of options passed to the plot of the time series values.
        figure_options : dict
            Dictionnary of options passed to plt.figure. Only used if ax is None.
        rc_Params_options: dict
            Dictionnary of options passed to plt.rcParams.update. Only used if ax is
            None.
        matplotlib_style: str
            Matplotlib style to be used. Only used if ax is None.

        Returns
        -------
        fig : matplotlib figure
            The resulting figure with S on its best match on X. A normalized
            shapelet will be scalled to macth the scale of X.

        """
        return self.transformer_vis.plot_on_X(
            id_shapelet,
            X,
            ax=ax,
            shp_scatter_options=shp_scatter_options,
            x_plot_options=x_plot_options,
            figure_options=figure_options,
            rc_Params_options=rc_Params_options,
            matplotlib_style=matplotlib_style,
        )

    def plot_distance_vector(
        self,
        id_shapelet,
        X,
        ax=None,
        show_legend=True,
        show_threshold=True,
        dist_plot_options={"linewidth": 2, "alpha": 0.9},  # noqa: B006
        threshold_plot_options={  # noqa: B006
            "linewidth": 2,
            "alpha": 0.9,
            "color": "purple",
            "label": "threshold",
        },
        figure_options={  # noqa: B006
            "figsize": (10, 5),
            "dpi": 100,
        },
        rc_Params_options={"font.size": 22},  # noqa: B006
        matplotlib_style="seaborn-v0_8",
    ):
        """
        Plot the shapelet distance vector computed between itself and X.

        Parameters
        ----------
        id_shapelet : int
            ID of the shapelet to plot.
        X : array, shape=(n_timestamps) or shape=(n_features, n_timestamps)
            Input time series
        ax : matplotlib axe
            A matplotlib axe on which to plot the figure. The default is None
            and will create a new figure of size figsize.
        show_legend : bool, optional
            Wheter to show legend. Default is True
        show_threshold: bool, optional
            Wheter to show threshold (if it is not set to None). Default is True.
        threshold_plot_options : dict
            Dictionnary of options passed to the line plot of the threshold.
        dist_plot_options : dict
            Dictionnary of options passed to the plot of the distance vector values.
        figure_options : dict
            Dictionnary of options passed to plt.figure. Only used if ax is None.
        rc_Params_options: dict
            Dictionnary of options passed to plt.rcParams.update. Only used if ax is
            None.
        matplotlib_style: str
            Matplotlib style to be used. Only used if ax is None.

        Returns
        -------
        fig : matplotlib figure
            The resulting figure with the distance vector obtained by d(S,X)

        """
        return self.transformer_vis.plot_distance_vector(
            id_shapelet,
            X,
            ax=ax,
            show_legend=show_legend,
            show_threshold=show_threshold,
            threshold_plot_options=threshold_plot_options,
            dist_plot_options=dist_plot_options,
            figure_options=figure_options,
            rc_Params_options=rc_Params_options,
            matplotlib_style=matplotlib_style,
        )

    def plot(
        self,
        id_shapelet,
        ax=None,
        scatter_options={  # noqa: B006
            "s": 70,
            "alpha": 0.75,
            "zorder": 3,
            "edgecolor": "black",
            "linewidths": 2,
        },
        plot_options={  # noqa: B006
            "linewidth": 2,
            "alpha": 0.9,
            "linestyle": "--",
        },
        figure_options={  # noqa: B006
            "figsize": (10, 5),
            "dpi": 100,
        },
        rc_Params_options={"font.size": 22},  # noqa: B006
        matplotlib_style="seaborn-v0_8",
        custom_title_string=None,
    ):
        """
        Plot the shapelet values.

        Parameters
        ----------
        id_shapelet : int
            ID of the shapelet to plot.
        ax : matplotlib axe
            A matplotlib axe on which to plot the figure. The default is None
            and will create a new figure of size figsize.
         scatter_options : dict
             Options to apply to scatter plot of the shapelet values.
         figure_options : dict
             Dictionnary of options passed to plt.figure. Only used if ax is None.
        figure_options : dict
            Dictionnary of options passed to plt.figure. Only used if ax is None.
        rc_Params_options: dict
            Dictionnary of options passed to plt.rcParams.update. Only used if ax is
            None.
        matplotlib_style: str
            Matplotlib style to be used. Only used if ax is None.
        custom_title_string : str
            If not None, use this string as title for the plot instead of the default
            one based on the shapelet parametres.

        Returns
        -------
        fig : matplotlib figure
            The resulting figure
        """
        return self.transformer_vis.plot(
            id_shapelet,
            ax=ax,
            plot_options=plot_options,
            scatter_options=scatter_options,
            figure_options=figure_options,
            rc_Params_options=rc_Params_options,
            matplotlib_style=matplotlib_style,
            custom_title_string=custom_title_string,
        )
