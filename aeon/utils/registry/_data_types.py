COLLECTIONS_DATA_TYPES = [
    "numpy3D",  # 3D np.ndarray of format (n_cases, n_channels, n_timepoints)
    "np-list",  # python list of 2D numpy array of length [n_cases],
    # each of shape (n_channels, n_timepoints_i)
    "df-list",  # python list of 2D pd.DataFrames of length [n_cases], each a of
    # shape (n_timepoints_i, n_channels)
    "numpy2D",  # 2D np.ndarray of shape (n_cases, n_timepoints)
    "pd-wide",  # 2D pd.DataFrame of shape (n_cases, n_timepoints)
    "nested_univ",  # pd.DataFrame (n_cases, n_channels) with each cell a pd.Series,
    "pd-multiindex",  # pd.DataFrame with multi-index,
]
