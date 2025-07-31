import numpy as np
from numba import njit, prange

# @njit(fastmath=True, cache=True)
# def numba_digitize(x, bins, right=False):
#     """
#     Numba implementation that produces identical output to np.digitize.
#     """
#     x_flat = x.flatten()
#     result = np.empty(x_flat.shape[0], dtype=np.intp)
    
#     for i in range(x_flat.shape[0]):
#         val = x_flat[i]
#         bin_idx = 0
        
#         if right:
#             # bins[i] < x <= bins[i+1]
#             for j in range(len(bins)):
#                 if val <= bins[j]:
#                     bin_idx = j
#                     break
#                 bin_idx = j + 1
#         else:
#             # bins[i] <= x < bins[i+1] (default behavior)
#             for j in range(len(bins)):
#                 if val < bins[j]:
#                     bin_idx = j
#                     break
#                 bin_idx = j + 1
        
#         result[i] = bin_idx
    
#     return result.reshape(x.shape)

@njit(fastmath=True, cache=True, parallel=True)
def numba_digitize_parallel(x, bins, right=False):
    """
    Parallel version for better performance on large arrays.
    """
    x_flat = x.flatten()
    result = np.empty(x_flat.shape[0], dtype=np.intp)
    
    for i in prange(x_flat.shape[0]):
        val = x_flat[i]
        bin_idx = 0
        
        if right:
            for j in range(len(bins)):
                if val <= bins[j]:
                    bin_idx = j
                    break
                bin_idx = j + 1
        else:
            for j in range(len(bins)):
                if val < bins[j]:
                    bin_idx = j
                    break
                bin_idx = j + 1
        
        result[i] = bin_idx
    
    return result.reshape(x.shape)


@njit(fastmath=True, cache=True, parallel=True)
def _parallel_get_sax_symbols(X, breakpoints):
    n_cases, n_channels, n_timepoints = X.shape
    X_new = np.zeros((n_cases, n_channels, n_timepoints), dtype=np.intp)
    n_break = breakpoints.shape[0] - 1
    for i_x in prange(n_cases):
        for i_c in prange(n_channels):
            for i_b in prange(n_break):
                mask = np.where(
                    (X[i_x, i_c] >= breakpoints[i_b])
                    & (X[i_x, i_c] < breakpoints[i_b + 1])
                )[0]
                X_new[i_x, i_c, mask] += np.array(i_b).astype(np.intp)

    return X_new


# Test to verify identical output
if __name__ == "__main__":
    x = np.array([[[0.2, 6.4, 3.0, 1.6]]])
    bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
    
    print("Original:", np.digitize(x, bins))
    print("Numba:   ", numba_digitize_parallel(x, bins))
    print("Match:   ", np.array_equal(np.digitize(x, bins), numba_digitize_parallel(x, bins)))

    print("Curr: ", _parallel_get_sax_symbols(x, bins))
    
    # Test with right=True
    print("\nWith right=True:")
    print("Original:", np.digitize(x, bins, right=True))
    print("Numba:   ", numba_digitize_parallel(x, bins, right=True))
    print("Match:   ", np.array_equal(np.digitize(x, bins, right=True), numba_digitize_parallel(x, bins, right=True)))
    print("Curr: ", _parallel_get_sax_symbols(x, bins))