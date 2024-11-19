"""Function to createthe Multi-Comparison Matrix (MCM) results visualisation."""

__maintainer__ = ["TonyBagnall"]

__all__ = ["create_multi_comparison_matrix"]

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from aeon.utils.validation._dependencies import _check_soft_dependencies


def create_multi_comparison_matrix(
    df_results,
    output_dir="./",
    pdf_savename=None,
    png_savename=None,
    csv_savename=None,
    tex_savename=None,
    used_statistic="Accuracy",
    save_as_json=False,
    plot_1v1_comparisons=False,
    order_win_tie_loss="higher",
    include_pvalue=True,
    pvalue_test="wilcoxon",
    pvalue_test_params=None,
    pvalue_correction=None,
    pvalue_threshold=0.05,
    use_mean="mean-difference",
    order_stats="average-statistic",
    order_better="decreasing",
    dataset_column=None,
    precision=4,
    load_analysis=False,
    row_comparates=None,
    col_comparates=None,
    excluded_row_comparates=None,
    excluded_col_comparates=None,
    colormap="coolwarm",
    fig_size="auto",
    font_size="auto",
    colorbar_orientation="vertical",
    colorbar_value=None,
    win_label="r>c",
    tie_label="r=c",
    loss_label="r<c",
    include_legend=True,
    show_symetry=True,
):
    """Generate the Multi-Comparison Matrix (MCM) [1]_.

    MCM summarises a set of results for multiple estimators evaluated on multiple
    datasets. The MCM is a heatmap that shows absolute performance and tests for
    significant difference. It is configurable inmany ways.

    Parameters
    ----------
    df_results: str or pd.DataFrame
        A csv file containing results in `n_problems,n_estimators` format. The first
        row should contain the names of the estimators and the first column can
        contain the names of the problems if `dataset_column` is true.
    output_dir: str, default = './'
        The output directory for the results.
    pdf_savename: str, default = None
        The name of the saved file into pdf format. if None, it will not be saved into
        this format.
    png_savename: str, default = None
        The name of the saved file into png format, if None, it will not be saved
        into this format.
    csv_savename: str, default = None
        The name of the saved file into csv format, if None, will not be saved into
        this format.
    tex_savename: str, default = None
        The name of the saved file into tex format, if None, will not be saved into
        this format.
    used_statistic: str, default = 'Score'
        Name of the metric being assesses (e.g. accuracy, error, mse).
    save_as_json: bool, default = True
        Whether or not to save the python analysis dict into a json file format.
    plot_1v1_comparisons: bool, default = True
        Whether or not to plot the 1v1 scatter results.
    order_win_tie_loss: str, default = 'higher'
        The order on considering a win or a loss for a given statistics.
    include_pvalue bool, default = True
        Condition whether or not include a pvalue stats.
    pvalue_test: str, default = 'wilcoxon'
        The statistical test to produce the pvalue stats. Currently only wilcoxon is
        supported.
    pvalue_test_params: dict, default = None,
        The default parameter set for the pvalue_test used. If pvalue_test is set
        to Wilcoxon, one should check the scipy.stats.wilcoxon parameters,
        in the case Wilcoxon is set and this parameter is None, then the default setup
        is {"zero_method": "pratt", "alternative": "greater"}.
    pvalue_correction: str, default = None
        Correction to use for the pvalue significant test, None or "Holm".
    pvalue_threshold: float, default = 0.05
        Threshold for considering a comparison is significant or not. If pvalue <
        pvalue_threshhold -> comparison is significant.
    use_mean: str, default = 'mean-difference'
        The mean used to compare two estimators. The only option available
        is 'mean-difference' which is the difference between arithmetic mean
        over all datasets.
    order_stats: str, default = 'average-statistic'
        The way to order the used_statistic, default setup orders by average
        statistic over all datasets.
        The options are:
        ===============================================================
        method                               what it does
        ===============================================================
        average-statistic      average used_statistic over all datasets
        average-rank           average rank over all datasets
        max-wins               maximum number of wins over all datasets
        amean-amean            average over difference of use_mean
        pvalue                 average pvalue over all comparates
        ================================================================
    order_better: str, default = 'decreasing'
        By which order to sort stats, from best to worse.
    dataset_column: str, default = 'dataset_name'
        The name of the datasets column in the csv file.
    precision: int, default = 4
        The number of floating numbers after decimal point.
    load_analysis: bool, default = False
        If True attempts to load the analysis json file.
    row_comparates: list of str, default = None
      A list of included row comparates, if None, all of the comparates in the study
      are placed in the rows.
    col_comparates: list of str, default = None
        A list of included col comparates, if None, all of the comparates in the
        study are placed in the cols.
    excluded_row_comparates: list of str, default = None
        A list of excluded row comparates. If None, all comparates are included.
    excluded_col_comparates: list of str, default = None
        A list of excluded col comparates. If None, all comparates are included.
    colormap: str, default = 'coolwarm'
        The colormap used in matplotlib, if set to None, no color map is used and the
        heatmap is turned off, no colors will be seen.
    fig_size: str ot tuple of two int, default = 'auto'
        The height and width of the figure, if 'auto', use _get_fig_size function in
        utils.py. Note that the fig size values are in matplotlib units.
    font_size: int, default = 17
        The font size of text.
    colorbar_orientation: str, default = 'vertical'
        In which orientation to show the colorbar either horizontal or vertical.
    colorbar_value: str, default = 'mean-difference'
        The values for which the heat map colors are based on.
    win_label: str, default = "r>c"
        The winning label to be set on the MCM.
    tie_label: str, default = "r=c"
        The tie label to be set on the MCM.
    loss_label: str, default = "r<c"
        The loss label to be set on the MCM.
    include_legend: bool, default = True
        Whether or not to show the legend on the MCM.
    show_symetry: bool, default = True
        Whether or not to show the symetrical part of the heatmap.

    Returns
    -------
    fig: plt.Figure
        The figure object of the heatmap.

    Example
    -------
    >>> from aeon.visualisation import create_multi_comparison_matrix # doctest: +SKIP
    >>> create_multi_comparison_matrix(df_results='results.csv') # doctest: +SKIP

    Notes
    -----
    Developed from the code in https://github.com/MSD-IRIMAS/Multi_Comparison_Matrix

    References
    ----------
    .. [1] Ismail-Fawaz A. et al, An Approach To Multiple Comparison Benchmark
    Evaluations That Is Stable Under Manipulation Of The Comparate Set
    arXiv preprint arXiv:2305.11921, 2023.
    """
    if isinstance(df_results, str):
        try:
            df_results = pd.read_csv(df_results)
        except Exception as e:
            raise ValueError(f"No dataframe or valid path is given: Exception {e}")

    analysis = _get_analysis(
        df_results,
        output_dir=output_dir,
        used_statistic=used_statistic,
        save_as_json=save_as_json,
        plot_1v1_comparisons=plot_1v1_comparisons,
        order_win_tie_loss=order_win_tie_loss,
        include_pvalue=include_pvalue,
        pvalue_test=pvalue_test,
        pvalue_test_params=pvalue_test_params,
        pvalue_correction=pvalue_correction,
        pvalue_threshhold=pvalue_threshold,
        use_mean=use_mean,
        order_stats=order_stats,
        order_better=order_better,
        dataset_column=dataset_column,
        precision=precision,
        load_analysis=load_analysis,
    )

    # start drawing heatmap
    temp = _draw(
        analysis,
        pdf_savename=pdf_savename,
        png_savename=png_savename,
        tex_savename=tex_savename,
        csv_savename=csv_savename,
        output_dir=output_dir,
        row_comparates=row_comparates,
        col_comparates=col_comparates,
        excluded_row_comparates=excluded_row_comparates,
        excluded_col_comparates=excluded_col_comparates,
        precision=precision,
        colormap=colormap,
        fig_size=fig_size,
        font_size=font_size,
        colorbar_orientation=colorbar_orientation,
        colorbar_value=colorbar_value,
        win_label=win_label,
        tie_label=tie_label,
        loss_label=loss_label,
        include_legend=include_legend,
        show_symetry=show_symetry,
    )
    return temp


def _get_analysis(
    df_results,
    output_dir="./",
    used_statistic="Score",
    save_as_json=True,
    plot_1v1_comparisons=False,
    order_win_tie_loss="higher",
    include_pvalue=True,
    pvalue_test="wilcoxon",
    pvalue_test_params=None,
    pvalue_correction=None,
    pvalue_threshhold=0.05,
    use_mean="mean-difference",
    order_stats="average-statistic",
    order_better="decreasing",
    dataset_column=None,
    precision=4,
    load_analysis=False,
):
    _check_soft_dependencies("matplotlib")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    def _plot_1v1(
        x,
        y,
        name_x,
        name_y,
        win_x,
        loss_x,
        tie,
        output_directory="./",
        min_lim: int = 0,
        max_lim: int = 1,
        scatter_size: int = 100,
        linewidth: int = 3,
        linecolor: str = "black",
        fontsize: int = 20,
    ):
        save_path = os.path.join(
            output_directory,
            "1v1_plots",
            _get_keys_for_two_comparates(name_x, name_y) + ".pdf",
        )
        if os.path.exists(save_path):
            return

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i in range(len(x)):
            if x[i] > y[i]:
                ax.scatter(y[i], x[i], color="blue", s=scatter_size)
            elif x[i] < y[i]:
                ax.scatter(y[i], x[i], color="orange", s=scatter_size)
            else:
                ax.scatter(y[i], x[i], color="green", s=scatter_size)

        ax.plot([min_lim, max_lim], [min_lim, max_lim], lw=linewidth, color=linecolor)
        ax.set_xlim(min_lim, max_lim)
        ax.set_ylim(min_lim, max_lim)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel(name_y, fontsize=fontsize)
        ax.set_ylabel(name_x, fontsize=fontsize)

        if pvalue_test == "wilcoxon":
            _pvalue_test_params = {}
            if pvalue_test_params is None:
                _pvalue_test_params = {
                    "zero_method": "wilcox",
                    "alternative": "greater",
                }
            else:
                _pvalue_test_params = pvalue_test_params
            p_value = round(wilcoxon(x=x, y=y, **_pvalue_test_params)[1], precision)
        else:
            raise ValueError("The test " + pvalue_test + " is not yet supported.")

        legend_elements = [
            mpl.lines.Line2D(
                [], [], marker="o", color="blue", label=f"Win {win_x}", linestyle="None"
            ),
            mpl.lines.Line2D(
                [], [], marker="o", color="green", label=f"Tie {tie}", linestyle="None"
            ),
            mpl.lines.Line2D(
                [],
                [],
                marker="o",
                color="orange",
                label=f"Loss {loss_x}",
                linestyle="None",
            ),
            mpl.lines.Line2D(
                [], [], marker=" ", color="none", label=f"P-Value {p_value}"
            ),
        ]

        ax.legend(handles=legend_elements)

        if not os.path.exists(output_directory + "1v1_plots/"):
            os.mkdir(output_directory + "1v1_plots/")
        plt.savefig(save_path, bbox_inches="tight")
        plt.savefig(save_path.replace(".pdf", ".png"), bbox_inches="tight")
        plt.cla()
        plt.clf()
        plt.close()

    save_file = output_dir + "analysis.json"

    if load_analysis and os.path.exists(save_file):
        with open(save_file) as json_file:
            analysis = json.load(json_file)

        return analysis

    analysis = {
        "dataset-column": dataset_column,
        "use-mean": use_mean,
        "order-stats": order_stats,
        "order-better": order_better,
        "used-statistics": used_statistic,
        "order-win_tie_loss": order_win_tie_loss,
        "include-pvalue": include_pvalue,
        "pvalue-test": pvalue_test,
        "pvalue-threshold": pvalue_threshhold,
        "pvalue-correction": pvalue_correction,
    }

    _decode_results_data_frame(df=df_results, analysis=analysis)

    if order_stats == "average-statistic":
        average_statistic = {}

    for i in range(analysis["n-comparates"]):
        comparate_i = analysis["comparate-names"][i]

        if order_stats == "average-statistic":
            average_statistic[comparate_i] = round(
                np.mean(df_results[comparate_i]), precision
            )

        for j in range(analysis["n-comparates"]):
            if i != j:
                comparate_j = analysis["comparate-names"][j]

                pairwise_key = _get_keys_for_two_comparates(comparate_i, comparate_j)

                x = df_results[comparate_i]
                y = df_results[comparate_j]

                pairwise_content = _get_pairwise_content(
                    x=x,
                    y=y,
                    order_WinTieLoss=order_win_tie_loss,
                    include_pvalue=include_pvalue,
                    pvalue_test=pvalue_test,
                    pvalue_test_params=pvalue_test_params,
                    pvalue_threshhold=pvalue_threshhold,
                    use_mean=use_mean,
                )

                analysis[pairwise_key] = pairwise_content

                if plot_1v1_comparisons:
                    max_lim = max(x.max(), y.max())
                    min_lim = min(x.min(), y.min())

                    if max_lim < 1:
                        max_lim = 1
                        min_lim = 0

                    _plot_1v1(
                        x=x,
                        y=y,
                        name_x=comparate_i,
                        name_y=comparate_j,
                        win_x=pairwise_content["win"],
                        tie=pairwise_content["tie"],
                        loss_x=pairwise_content["loss"],
                        output_directory=output_dir,
                        max_lim=max_lim,
                        min_lim=min_lim,
                    )

    if order_stats == "average-statistic":
        analysis["average-statistic"] = average_statistic

    if pvalue_correction == "Holm":
        _holms_correction(analysis=analysis)

    _re_order_comparates(df_results=df_results, analysis=analysis)

    if save_as_json:
        with open(save_file, "w") as fjson:
            json.dump(analysis, fjson, cls=_NpEncoder)

    return analysis


def _draw(
    analysis,
    output_dir="./",
    pdf_savename=None,
    png_savename=None,
    csv_savename=None,
    tex_savename=None,
    row_comparates=None,
    col_comparates=None,
    excluded_row_comparates=None,
    excluded_col_comparates=None,
    precision=4,
    colormap="coolwarm",
    fig_size="auto",
    font_size="auto",
    colorbar_orientation="vertical",
    colorbar_value=None,
    win_label="r>c",
    tie_label="r=c",
    loss_label="r<c",
    show_symetry=True,
    include_legend=True,
):
    _check_soft_dependencies("matplotlib")
    import matplotlib.pyplot as plt

    latex_string = "\\documentclass[a4,12pt]{article}\n"
    latex_string += "\\usepackage{colortbl}\n"
    latex_string += "\\usepackage{pgfplots}\n"
    latex_string += "\\usepackage[margin=2cm]{geometry}\n"
    latex_string += "\\pgfplotsset{compat=newest}\n"
    latex_string += "\\begin{document}\n"
    latex_string += "\\begin{table}\n"
    latex_string += "\\footnotesize\n"
    latex_string += "\\sffamily\n"
    latex_string += "\\begin{center}\n"

    if (col_comparates is not None) and (excluded_col_comparates is not None):
        raise ValueError("Choose whether to include or exclude, not both")

    if (row_comparates is not None) and (excluded_row_comparates is not None):
        raise ValueError("Choose whether to include or exclude, not both")

    if row_comparates is None:
        row_comparates = analysis["ordered-comparate-names"]
    else:
        # order comparates
        row_comparates = [
            x for x in analysis["ordered-comparate-names"] if x in row_comparates
        ]

    if col_comparates is None:
        col_comparates = analysis["ordered-comparate-names"]
    else:
        col_comparates = [
            x for x in analysis["ordered-comparate-names"] if x in col_comparates
        ]

    if excluded_row_comparates is not None:
        row_comparates = [
            x
            for x in analysis["ordered-comparate-names"]
            if not (x in excluded_row_comparates)
        ]

    if excluded_col_comparates is not None:
        col_comparates = [
            x
            for x in analysis["ordered-comparate-names"]
            if not (x in excluded_col_comparates)
        ]

    n_rows = len(row_comparates)
    n_cols = len(col_comparates)

    can_be_symmetrical = False

    if n_rows == n_cols == len(analysis["ordered-comparate-names"]):
        can_be_symmetrical = True

    if n_rows == n_cols == 1:
        figure_aspect = "equal"
        colormap = None

        if row_comparates[0] == col_comparates[0]:
            raise ValueError(
                f"Row and Column comparates are the same" f" {row_comparates[0]}"
            )
            return
    else:
        figure_aspect = "auto"

    if (n_rows == 1) and (n_cols == 2):
        colorbar_orientation = "horizontal"

    elif (n_rows == 2) and (n_cols == 2):
        colorbar_orientation = "vertical"

    elif (n_rows == 2) and (n_cols == 1):
        colorbar_orientation = "vertical"

    elif n_rows <= 2:
        colorbar_orientation = "horizontal"

    if include_legend:
        cell_legend, longest_string = _get_cell_legend(
            analysis, win_label=win_label, tie_label=tie_label, loss_label=loss_label
        )

        if analysis["include-pvalue"]:
            p_value_text = (
                f"If in bold, then\np-value < {analysis['pvalue-threshold']:.2f}"
            )

            if analysis["pvalue-correction"] is not None:
                correction = _capitalize_label(analysis["pvalue-correction"])
                p_value_text = f"{p_value_text}\n{correction} correction"

        else:
            p_value_text = ""

        longest_string = max(longest_string, len(p_value_text))

    else:
        cell_legend = ""
        p_value_text = ""
        longest_string = len(f"{win_label} / {tie_label} / {loss_label}")

    annot_out = _get_annotation(
        analysis=analysis,
        row_comparates=row_comparates,
        col_comparates=col_comparates,
        cell_legend=cell_legend,
        p_value_text=p_value_text,
        colormap=colormap,
        colorbar_value=colorbar_value,
        precision=precision,
    )

    df_annotations = annot_out["df_annotations"]
    pairwise_matrix = annot_out["pairwise_matrix"]

    n_info_per_cell = annot_out["n_info_per_cell"]

    legend_cell_location = annot_out["legend_cell_location"]
    p_value_cell_location = annot_out["p_value_cell_location"]

    longest_string = max(annot_out["longest_string"], longest_string)

    if csv_savename is not None:
        # todo: can add a argument to save or not
        df_annotations.to_csv(output_dir + f"{csv_savename}.csv", index=False)

    df_annotations.drop("comparates", inplace=True, axis=1)
    df_annotations_np = np.asarray(df_annotations)

    figsize = _get_fig_size(
        fig_size=fig_size,
        n_rows=n_rows,
        n_cols=n_cols,
        n_info_per_cell=n_info_per_cell,
        longest_string=longest_string,
    )

    if font_size == "auto":
        if (n_rows <= 2) and (n_cols <= 2):
            font_size = 8
        else:
            font_size = 10

    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]))
    ax.grid(False)

    _can_be_negative = False
    if colorbar_value is None or colorbar_value == "mean-difference":
        _can_be_negative = True
    min_value, max_value = _get_limits(
        pairwise_matrix=pairwise_matrix, can_be_negative=_can_be_negative
    )

    if colormap is None:
        _colormap = "coolwarm"
        _vmin, _vmax = -2, 2
    else:
        _colormap = colormap
        _vmin = min_value + 0.2 * min_value
        _vmax = max_value + 0.2 * max_value

    if colorbar_value is None:
        _colorbar_value = _capitalize_label("mean-difference")
    else:
        _colorbar_value = _capitalize_label(colorbar_value)

    im = ax.imshow(
        pairwise_matrix, cmap=colormap, aspect=figure_aspect, vmin=_vmin, vmax=_vmax
    )

    if colormap is not None:
        if (
            (p_value_cell_location is None)
            and (legend_cell_location is None)
            and (colorbar_orientation == "horizontal")
        ):
            shrink = 0.4
        else:
            shrink = 0.5

        cbar = ax.figure.colorbar(
            im, ax=ax, shrink=shrink, orientation=colorbar_orientation
        )
        cbar.ax.tick_params(labelsize=font_size)
        cbar.set_label(label=_capitalize_label(_colorbar_value), size=font_size)

    cm_norm = plt.Normalize(_vmin, _vmax)
    cm = plt.colormaps[_colormap]

    xticks, yticks = _get_ticks(analysis, row_comparates, col_comparates, precision)
    ax.set_xticks(np.arange(n_cols), labels=xticks, fontsize=font_size)
    ax.set_yticks(np.arange(n_rows), labels=yticks, fontsize=font_size)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    ax.spines[:].set_visible(False)

    start_j = 0

    if analysis["order-stats"] == "average-statistic":
        ordering = "Mean-" + analysis["used-statistics"]
    else:
        ordering = analysis["order-stats"]

    latex_table = []
    latex_table.append(
        [f"{ordering}"]
        + [rf"\shortstack{{{_}}}".replace("\n", " \\\\ ") for _ in xticks]
    )

    for i in range(n_rows):
        row_comparate = row_comparates[i]

        latex_row = []

        if can_be_symmetrical and (not show_symetry):
            start_j = i

        for j in range(start_j, n_cols):
            col_comparate = col_comparates[j]

            cell_text_arguments = dict(
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=font_size,
            )

            if row_comparate == col_comparate:
                if p_value_cell_location is not None:
                    if (i == p_value_cell_location[0]) and (
                        j == p_value_cell_location[1]
                    ):
                        cell_text_arguments.update(
                            fontweight="bold", fontsize=font_size
                        )

                if legend_cell_location is not None:
                    if (i == legend_cell_location[0]) and (
                        j == legend_cell_location[1]
                    ):
                        cell_text_arguments.update(fontsize=font_size)

                im.axes.text(j, i, df_annotations_np[i, j], **cell_text_arguments)

                latex_cell = "\\rule{0em}{3ex} " + df_annotations_np[i, j].replace(
                    "\n", " \\\\ "
                )
                r = [
                    str(round(_, precision))
                    for _ in cm(cm_norm(pairwise_matrix[i, j]))[:-1]
                ]
                latex_row.append(
                    f"\\cellcolor[rgb]{{{','.join(r)}}}\\shortstack{{{latex_cell}}}"
                )

                continue

            pairwise_key = _get_keys_for_two_comparates(row_comparate, col_comparate)

            pairwise_content = analysis[pairwise_key]
            pairwise_keys = list(pairwise_content.keys())

            latex_bold = ""

            if "pvalue" in pairwise_keys:
                if analysis[pairwise_key]["is-significant"]:
                    cell_text_arguments.update(fontweight="bold")
                    latex_bold = "\\bfseries "

            im.axes.text(j, i, df_annotations_np[i, j], **cell_text_arguments)

            latex_cell = "\\rule{0em}{3ex} " + df_annotations_np[i, j].replace(
                "\n", " \\\\ "
            )
            s1 = f"{latex_bold}\\cellcolor[rgb]"
            s2 = [
                str(round(_, precision))
                for _ in cm(cm_norm(pairwise_matrix[i, j]))[:-1]
            ]
            s = f"{s1}{{{','.join(s2)}}}\\shortstack{{{latex_cell}}}"
            latex_row.append(s)

        if legend_cell_location is None:
            latex_cell = (
                "\\rule{0em}{3ex} " + f"{cell_legend}".replace("\n", " \\\\ ")
                if i == 0
                else "\\null"
            )
            latex_row.append(f"\\shortstack{{{latex_cell}}}")

        latex_table.append(
            [rf"\shortstack{{{yticks[i]}}}".replace("\n", " \\\\ ")] + latex_row
        )

    if n_cols == n_rows == 1:
        # special case when 1x1
        x = ax.get_position().x0 - 1
        y = ax.get_position().y1 - 1.5
    else:
        x = ax.get_position().x0 - 0.8
        y = ax.get_position().y1 - 1.5

    im.axes.text(
        x,
        y,
        ordering,
        fontsize=font_size,
        horizontalalignment="center",
        verticalalignment="center",
    )

    if p_value_cell_location is None:
        x = 0
        y = n_rows

        if n_rows == n_cols == 1:
            y = 0.7
        elif (n_cols == 1) and (legend_cell_location is None):
            x = -0.5
        elif (n_rows == 1) and (n_cols <= 2) and (colorbar_orientation == "horizontal"):
            x = -0.5
        im.axes.text(
            x,
            y,
            p_value_text,
            fontsize=font_size,
            fontweight="bold",
            horizontalalignment="center",
            verticalalignment="center",
        )

    if legend_cell_location is None:
        x = n_cols - 1
        y = n_rows
        if n_rows == n_cols == 1:
            x = n_cols + 0.5
            y = 0

        elif (n_rows == 1) and (colorbar_orientation == "horizontal"):
            x = n_cols + 0.25
            y = 0

        elif n_cols == 1:
            x = 0.5

        im.axes.text(
            x,
            y,
            cell_legend,
            fontsize=font_size,
            horizontalalignment="center",
            verticalalignment="center",
        )

    if pdf_savename is not None:
        plt.savefig(
            os.path.join(output_dir + f"{pdf_savename}.pdf"), bbox_inches="tight"
        )
        plt.cla()
        plt.clf()
        plt.close()
    elif png_savename is not None:
        plt.savefig(
            os.path.join(output_dir + f"{png_savename}.png"), bbox_inches="tight"
        )
        plt.cla()
        plt.clf()

    latex_string += (
        f"\\begin{{tabular}}{{{'c' * (len(latex_table[0]) + 1)}}}\n"  # +1 for labels
    )
    for latex_row in latex_table:
        latex_string += " & ".join(latex_row) + " \\\\[1ex]" + "\n"

    if colorbar_orientation == "horizontal":
        latex_string += "\\end{tabular}\\\\\n"
    else:
        latex_string += "\\end{tabular}\n"

    latex_colorbar_0 = (
        "\\begin{tikzpicture}[baseline=(current bounding box.center)]"
        "\\begin{axis}[hide axis,scale only axis,"
    )
    t1 = [str(int(_ * 255)) for _ in cm(cm_norm(min_value))[:-1]]
    t2 = [str(int(_ * 255)) for _ in cm(cm_norm(max_value))[:-1]]
    latex_colorbar_1 = (
        f"colormap={{cm}}{{rgb255(1)=({','.join(t1)}) rgb255(2)=(220,"
        f"220,220) rgb255(3)=({','.join(t2)})}},"
    )
    latex_colorbar_2 = (
        f"colorbar horizontal,point meta min={_vmin:.02f},point meta max={_vmax:.02f},"
    )
    latex_colorbar_3 = "colorbar/width=1.0em"
    latex_colorbar_4 = "}] \\addplot[_draw=none] {0};\\end{axis}\\end{tikzpicture}"

    if colorbar_orientation == "horizontal":
        latex_string += (
            latex_colorbar_0 + r"width=0sp,height=0sp,colorbar horizontal,colorbar "
            r"style={width=0.25\linewidth,"
            + latex_colorbar_1
            + latex_colorbar_2
            + latex_colorbar_3
            + ",scaled x ticks=false,xticklabel style={/pgf/number "
            "format/fixed,/pgf/number format/precision=3},"
            + f"xlabel={{{_colorbar_value}}},"
            + latex_colorbar_4
        )
    else:
        latex_string += (
            latex_colorbar_0
            + r"width=1pt,colorbar right,colorbar style={height=0.25\linewidth,"
            + latex_colorbar_1
            + latex_colorbar_2
            + latex_colorbar_3
            + ",scaled y ticks=false,ylabel style={rotate=180},yticklabel "
            "style={/pgf/number format/fixed,/pgf/number format/precision=3},"
            + f"ylabel={{{_colorbar_value}}},"
            + latex_colorbar_4
        )

    latex_string += "\\end{center}\n"
    latex_string += (
        "\\caption{[...] \\textbf{"
        + f"{p_value_text}".replace("\n", " ")
        + "} [...]}\n"
    )
    latex_string += "\\end{table}\n"
    latex_string += "\\end{document}\n"

    latex_string = latex_string.replace(">", "$>$")
    latex_string = latex_string.replace("<", "$<$")

    if tex_savename is not None:
        with open(
            f"{output_dir}/{tex_savename}.tex", "w", encoding="utf8", newline="\n"
        ) as file:
            file.writelines(latex_string)

    # latex references:
    # * https://tex.stackexchange.com/a/120187
    # * https://tex.stackexchange.com/a/334293
    # * https://tex.stackexchange.com/a/592942
    # * https://tex.stackexchange.com/a/304215
    return plt.Figure()


def _get_keys_for_two_comparates(a, b):
    return f"{a}-vs-{b}"


def _decode_results_data_frame(df, analysis):
    """Decode the necessary information from the DataFrame and into the json file.

    Parameters
    ----------
    df          : pandas DataFrame
        DF Containing the statistics of each comparate on multiple datasets.
        shape = (n_datasets, n_comparates), columns = [list of comparates names]
    analysis    : python dictionary

    """
    df_columns = list(df.columns)  # extract columns from data frame

    # check if dataset column name is correct

    if analysis["dataset-column"] is not None:
        if analysis["dataset-column"] not in df_columns:
            raise KeyError("The column " + analysis["dataset-column"] + " is missing.")

    # get number of examples (datasets)
    # n_datasets = len(np.unique(np.asarray(df[analysis['dataset-column']])))
    n_datasets = len(df.index)

    analysis["n-datasets"] = n_datasets  # add number of examples to dictionary

    if analysis["dataset-column"] is not None:
        analysis["dataset-names"] = list(
            df[analysis["dataset-column"]]
        )  # add example names to dict
        df_columns.remove(
            analysis["dataset-column"]
        )  # drop the dataset column name from columns list
        # and keep comparate names

    comparate_names = df_columns.copy()
    n_comparates = len(comparate_names)

    # add the information about comparates to dict
    analysis["comparate-names"] = comparate_names
    analysis["n-comparates"] = n_comparates


def _get_pairwise_content(
    x,
    y,
    order_WinTieLoss="higher",
    include_pvalue=True,
    pvalue_test="wilcoxon",
    pvalue_test_params=None,
    pvalue_threshhold=0.05,
    use_mean="mean-difference",
):
    content = {}

    if order_WinTieLoss == "lower":
        win = len(x[x < y])
        loss = len(x[x > y])
        tie = len(x[x == y])

    else:  # by default we assume higher is better
        win = len(x[x > y])
        loss = len(x[x < y])
        tie = len(x[x == y])

    content["win"] = win
    content["tie"] = tie
    content["loss"] = loss

    if include_pvalue:
        if pvalue_test == "wilcoxon":
            _pvalue_test_params = {}
            if pvalue_test_params is None:
                _pvalue_test_params = {"zero_method": "pratt", "alternative": "greater"}
            else:
                _pvalue_test_params = pvalue_test_params
            pvalue = wilcoxon(x=x, y=y, **_pvalue_test_params)[1]
            content["pvalue"] = pvalue

            if pvalue_test == "wilcoxon":
                if pvalue < pvalue_threshhold:
                    content["is-significant"] = True
                else:
                    content["is-significant"] = False

            else:
                raise ValueError(f"{pvalue_test} test is not supported yet")
    if use_mean == "mean-difference":
        content["mean"] = np.mean(x) - np.mean(y)

    return content


def _holms_correction(analysis):
    pvalues = []

    for i in range(analysis["n-comparates"]):
        comparate_i = analysis["comparate-names"][i]

        for j in range(i + 1, analysis["n-comparates"]):
            if i != j:
                comparate_j = analysis["comparate-names"][j]
                pairwise_key = _get_keys_for_two_comparates(comparate_i, comparate_j)
                pvalues.append(analysis[pairwise_key]["pvalue"])

    pvalues_sorted = np.sort(pvalues)

    k = 0
    m = len(pvalues)

    pvalue_times_used = {}

    for pvalue in pvalues:
        pvalue_times_used[pvalue] = 0

    for i in range(analysis["n-comparates"]):
        comparate_i = analysis["comparate-names"][i]

        for j in range(i + 1, analysis["n-comparates"]):
            if i != j:
                comparate_j = analysis["comparate-names"][j]
                pairwise_key = _get_keys_for_two_comparates(comparate_i, comparate_j)
                pvalue = analysis[pairwise_key]["pvalue"]
                index_pvalue = np.where(pvalues_sorted == pvalue)[0]

                if len(index_pvalue) == 1:
                    index_pvalue = index_pvalue[0]
                else:
                    index_pvalue = index_pvalue[pvalue_times_used[pvalue]]
                    pvalue_times_used[pvalue] += 1

                pvalue_threshhold_corrected = analysis["pvalue-threshold"] / (
                    m - index_pvalue
                )

                if pvalue < pvalue_threshhold_corrected:
                    analysis[pairwise_key]["is-significant"] = True
                else:
                    analysis[pairwise_key]["is-significant"] = False

                k = k + 1

    for i in range(analysis["n-comparates"]):
        comparate_i = analysis["comparate-names"][i]

        for j in range(i + 1, analysis["n-comparates"]):
            comparate_j = analysis["comparate-names"][j]

            pairwise_key_ij = _get_keys_for_two_comparates(comparate_i, comparate_j)
            pairwise_key_ji = _get_keys_for_two_comparates(comparate_j, comparate_i)

            analysis[pairwise_key_ji]["is-significant"] = analysis[pairwise_key_ij][
                "is-significant"
            ]
            analysis[pairwise_key_ji]["pvalue"] = analysis[pairwise_key_ij]["pvalue"]


def _re_order_comparates(df_results, analysis):
    stats = []

    if analysis["order-stats"] == "average-statistic":
        for i in range(analysis["n-comparates"]):
            stats.append(analysis["average-statistic"][analysis["comparate-names"][i]])

    elif analysis["order-stats"] == "average-rank":
        if analysis["dataset-column"] is not None:
            np_results = np.asarray(
                df_results.drop([analysis["dataset-column"]], axis=1)
            )
        else:
            np_results = np.asarray(df_results)

        df = pd.DataFrame(columns=["comparate-name", "values"])

        for i, comparate_name in enumerate(analysis["comparate-names"]):
            for j in range(analysis["n-datasets"]):
                df = df.append(
                    {"comparate-name": comparate_name, "values": np_results[j][i]},
                    ignore_index=True,
                )

        rank_values = np.array(df["values"]).reshape(
            analysis["n-comparates"], analysis["n-datasets"]
        )
        df_ranks = pd.DataFrame(data=rank_values)

        average_ranks = df_ranks.rank(ascending=False).mean(axis=1)

        stats = np.asarray(average_ranks)

    elif analysis["order-stats"] == "max-wins":
        for i in range(analysis["n-comparates"]):
            wins = []

            for j in range(analysis["n-comparates"]):
                if i != j:
                    wins.append(
                        analysis[
                            analysis["comparate-names"][i]
                            + "-vs-"
                            + analysis["comparate-names"][j]
                        ]["win"]
                    )

            stats.append(int(np.max(wins)))

    elif analysis["order-stats"] == "amean-amean":
        for i in range(analysis["n-comparates"]):
            ameans = []

            for j in range(analysis["n-comparates"]):
                if i != j:
                    ameans.append(
                        analysis[
                            analysis["comparate-names"][i]
                            + "-vs-"
                            + analysis["comparate-names"][j]
                        ]["mean"]
                    )

            stats.append(np.mean(ameans))

    elif analysis["order-stats"] == "pvalue":
        for i in range(analysis["n-comparates"]):
            pvalues = []

            for j in range(analysis["n-comparates"]):
                if i != j:
                    pvalues.append(
                        analysis[
                            analysis["comparate-names"][i]
                            + "-vs-"
                            + analysis["comparate-names"][j]
                        ]["pvalue"]
                    )

            stats.append(np.mean(pvalues))

    if analysis["order-better"] == "increasing":
        ordered_indices = np.argsort(stats)
    else:  # decreasing
        ordered_indices = np.argsort(stats)[::-1]

    analysis["ordered-stats"] = list(np.asarray(stats)[ordered_indices])
    analysis["ordered-comparate-names"] = list(
        np.asarray(analysis["comparate-names"])[ordered_indices]
    )


def _get_cell_legend(
    analysis,
    win_label="r>c",
    tie_label="r=c",
    loss_label="r<c",
):
    cell_legend = _capitalize_label(analysis["use-mean"])
    longest_string = len(cell_legend)

    win_tie_loss_string = f"{win_label} / {tie_label} / {loss_label}"
    longest_string = max(longest_string, len(win_tie_loss_string))

    cell_legend = f"{cell_legend}\n{win_tie_loss_string}"

    if analysis["include-pvalue"]:
        longest_string = max(
            longest_string, len(_capitalize_label(analysis["pvalue-test"]))
        )
        pvalue_test = _capitalize_label(analysis["pvalue-test"]) + " p-value"
        cell_legend = f"{cell_legend}\n{pvalue_test}"

    return cell_legend, longest_string


def _capitalize_label(s):
    if len(s.split("-")) == 1:
        return s.capitalize()

    else:
        return "-".join(ss.capitalize() for ss in s.split("-"))


def _get_annotation(
    analysis,
    row_comparates,
    col_comparates,
    cell_legend,
    p_value_text,
    colormap="coolwarm",
    colorbar_value=None,
    precision=4,
):
    fmt = f".{precision}f"

    n_rows = len(row_comparates)
    n_cols = len(col_comparates)

    pairwise_matrix = np.zeros(shape=(n_rows, n_cols))

    df_annotations = []

    n_info_per_cell = 0
    longest_string = 0

    p_value_cell_location = None
    legend_cell_location = None

    for i in range(n_rows):
        row_comparate = row_comparates[i]
        dict_to_add = {"comparates": row_comparate}
        longest_string = max(longest_string, len(row_comparate))

        for j in range(n_cols):
            col_comparate = col_comparates[j]

            if row_comparate != col_comparate:
                longest_string = max(longest_string, len(col_comparate))
                pairwise_key = _get_keys_for_two_comparates(
                    row_comparate, col_comparate
                )

                if colormap is not None:
                    try:
                        pairwise_matrix[i, j] = analysis[pairwise_key][colorbar_value]
                    except Exception:
                        pairwise_matrix[i, j] = analysis[pairwise_key]["mean"]

                else:
                    pairwise_matrix[i, j] = 0

                pairwise_content = analysis[pairwise_key]
                pairwise_keys = list(pairwise_content.keys())

                string_in_cell = f"{pairwise_content['mean']:{fmt}}\n"
                n_info_per_cell = 1

                if "win" in pairwise_keys:
                    string_in_cell = f"{string_in_cell}{pairwise_content['win']} / "
                    string_in_cell = f"{string_in_cell}{pairwise_content['tie']} / "
                    string_in_cell = f"{string_in_cell}{pairwise_content['loss']}\n"

                    n_info_per_cell += 1

                if "p-x-wins" in pairwise_keys:
                    string_in_cell = (
                        f"{string_in_cell}{pairwise_content['p-x-wins']:{fmt}} / "
                    )
                    string_in_cell = (
                        f"{string_in_cell}{pairwise_content['p-rope']:{fmt}} / "
                    )
                    string_in_cell = (
                        f"{string_in_cell}{pairwise_content['p-y-wins']:{fmt}}\n"
                    )

                if "pvalue" in pairwise_keys:
                    _p_value = round(pairwise_content["pvalue"], precision)
                    alpha = 10 ** (-precision)

                    if _p_value < alpha:
                        string_in_cell = rf"{string_in_cell} $\leq$ {alpha:.0e}"
                    else:
                        string_in_cell = (
                            f"{string_in_cell}{pairwise_content['pvalue']:{fmt}}"
                        )

                    n_info_per_cell += 1

                dict_to_add[col_comparate] = string_in_cell

            else:
                if legend_cell_location is None:
                    dict_to_add[row_comparate] = cell_legend
                    legend_cell_location = (i, j)
                else:
                    dict_to_add[col_comparate] = "-"
                    p_value_cell_location = (i, j)

                pairwise_matrix[i, j] = 0.0

        df_annotations.append(dict_to_add)

    if p_value_cell_location is not None:
        col_comparate = col_comparates[p_value_cell_location[1]]
        df_annotations[p_value_cell_location[0]][col_comparate] = p_value_text

    df_annotations = pd.DataFrame(df_annotations)

    out = dict(
        df_annotations=df_annotations,
        pairwise_matrix=pairwise_matrix,
        n_info_per_cell=n_info_per_cell,
        longest_string=longest_string,
        legend_cell_location=legend_cell_location,
        p_value_cell_location=p_value_cell_location,
    )

    return out


def _get_fig_size(
    fig_size,
    n_rows,
    n_cols,
    n_info_per_cell=None,
    longest_string=None,
):
    if isinstance(fig_size, str):
        if fig_size == "auto":
            if (n_rows == 1) and (n_cols == 2):
                size = [
                    int(max(longest_string * 0.13, 1) * n_cols),
                    int(max(n_info_per_cell * 0.1, 1) * (n_rows + 1)),
                ]

            elif n_rows <= n_cols:
                size = [
                    int(max(longest_string * 0.125, 1) * n_cols),
                    int(max(n_info_per_cell * 0.1, 1) * (n_rows + 1)),
                ]

            else:
                size = [
                    int(max(longest_string * 0.1, 1) * (n_cols + 1)),
                    int(max(n_info_per_cell * 0.125, 1) * n_rows),
                ]

            if n_rows == n_cols == 1:
                size[0] = size[0] + int(longest_string * 0.125)

            return size

        return [int(s) for s in fig_size.split(",")]

    return fig_size


def _get_limits(pairwise_matrix, can_be_negative=False, precision=4):
    if pairwise_matrix.shape[0] == 1:
        min_value = round(np.min(pairwise_matrix), precision)
        max_value = round(np.max(pairwise_matrix), precision)

        if min_value >= 0 and max_value >= 0 and (not can_be_negative):
            return min_value, max_value

        return -max(abs(min_value), abs(max_value)), max(abs(min_value), abs(max_value))
    min_value = np.min(pairwise_matrix)
    max_value = np.max(pairwise_matrix)
    if min_value < 0 or max_value < 0:
        max_min_max = max(abs(min_value), abs(max_value))

        min_value = _get_sign(x=min_value) * max_min_max
        max_value = _get_sign(x=max_value) * max_min_max
    return round(min_value, precision), round(max_value, precision)


def _get_ticks(analysis, row_comparates, col_comparates, precision=4):
    fmt = f"{precision}f"
    xticks = []
    yticks = []

    n_rows = len(row_comparates)
    n_cols = len(col_comparates)

    all_comparates = analysis["ordered-comparate-names"]
    all_stats = analysis["ordered-stats"]

    for i in range(n_rows):
        stat = all_stats[
            [
                x
                for x in range(len(all_comparates))
                if all_comparates[x] == row_comparates[i]
            ][0]
        ]

        tick_label = f"{row_comparates[i]}\n{stat:.{fmt}}"
        yticks.append(tick_label)

    for i in range(n_cols):
        stat = all_stats[
            [
                x
                for x in range(len(all_comparates))
                if all_comparates[x] == col_comparates[i]
            ][0]
        ]
        tick_label = f"{col_comparates[i]}\n{stat:.{fmt}}"
        xticks.append(tick_label)

    return xticks, yticks


def _get_sign(x):
    return 1 if x > 0 else -1


class _NpEncoder(json.JSONEncoder):
    """Encoder for json files saving."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
