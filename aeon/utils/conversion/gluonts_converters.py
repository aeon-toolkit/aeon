"""Converter utilities for gluonts data structures."""

import pandas as pd

from aeon.utils.conversion import convert_collection
from aeon.utils.validation._dependencies import _check_soft_dependencies


def convert_from_multiindex_to_listdataset(trainDF, class_val_list=None):
    """
    Output a dataset in ListDataset format compatible with gluonts.

    Parameters
    ----------
    trainDF: Multiindex dataframe
        Input DF should be multi-index DataFrame.
        Time index must be absolute.
    class_val_list: str
        List of classes in case of classification dataset.
        If not available, class_val_list will show instance numbers

    Returns
    -------
    A ListDataset mtype type to be used as input for gluonts models/estimators

    """
    _check_soft_dependencies("gluonts", severity="error")

    # New dependency from Gluon-ts
    import numpy as np
    import pandas as pd
    from gluonts.dataset.common import ListDataset

    dimension_name = trainDF.columns
    n_channels = len(trainDF.columns)

    # Convert to nested_univ format
    trainDF = convert_collection(trainDF, output_type="nested_univ")
    trainDF = trainDF.reset_index()
    trainDF = trainDF[dimension_name]

    # Infer frequency
    # Frequency is inferred from pd.Series's index
    # All instances must have the same freq and only fixed freq is supported
    # For a list of acceptable freq, see
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    freq = pd.infer_freq(trainDF.loc[0, dimension_name[0]].index)

    # Find start date for each instance
    start_date = [
        str(trainDF.loc[instance, dimension_name[0]].index[0].date())
        for instance, dim in trainDF.iterrows()
    ]

    if class_val_list is not None:
        feat_static_cat = class_val_list
    else:
        # If not available, class_val_list will show instance numbers
        feat_static_cat = list(np.arange(len(trainDF)))
    if n_channels > 1:
        one_dim_target = False
    else:
        one_dim_target = True

    all_instance_list = []
    for instance, _dim_name in trainDF.iterrows():
        one_instance_list = []
        for dim in range(n_channels):
            tmp = list(trainDF.loc[instance, dimension_name[dim]].to_numpy())
            one_instance_list.append(tmp)
        if one_dim_target is True:
            flatlist = [element for sublist in one_instance_list for element in sublist]
            all_instance_list.append(flatlist)
        else:
            all_instance_list.append(one_instance_list)
    train_ds = ListDataset(
        [
            {"target": target, "start": start, "fea_static_cat": [fsc]}
            for (target, start, fsc) in zip(
                all_instance_list, start_date, feat_static_cat
            )
        ],
        freq=freq,
        one_dim_target=one_dim_target,
    )
    return train_ds


def convert_gluonts_result_to_multiindex(gluonts_result):
    """

    Back Convert from Gluonts to aeon.

    Convert the output of Gluonts's prediction to a multiindex
    dataframe compatible with aeon.

    Parameters
    ----------
    gluonts_result: The first element of the tuple resulting
    from running `make_evaluation_predictions`.
        For example in Eg:
        forecast_it, ts_it = make_evaluation_predictions()
        gluonts_result = forecast_it

    Returns
    -------
    A MultiIndex pandas DataFrame.
    """
    instance_no = len(gluonts_result)
    global_ls = []
    per_instance_ls = []
    columns = []

    for i in range(instance_no):
        validation_no = gluonts_result[i].samples.shape[0]
        period = gluonts_result[i].samples.shape[1]
        start_date = pd.to_datetime(gluonts_result[i].start_date)
        freq = gluonts_result[i].freq
        ts_index = pd.date_range(start=start_date, periods=period, freq=freq)
        per_instance_ls = [
            pd.Series(data=gluonts_result[i].samples[j], index=ts_index)
            for j in range(validation_no)
        ]
        global_ls.append(per_instance_ls)

    for k in range(validation_no):
        columns.append("validation_" + str(k))

    nested_univ = pd.DataFrame(global_ls, columns=columns)

    return convert_collection(nested_univ, output_type="pd-multiindex")
