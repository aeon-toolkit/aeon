def _get_min_length(X):
    min_length = X[0].shape[1]
    for x in X:
        if x.shape[1] < min_length:
            min_length = x.shape[1]
    return min_length


def _get_max_length(X):
    max_length = X[0].shape[1]
    for x in X:
        if x.shape[1] > max_length:
            max_length = x.shape[1]
    return max_length
