from numba import njit


@njit(cache=True)
def numba_func_1(x, y):
    return x + y

@njit(cache=True)
def numba_func2(x, y, test="joe"):
    print(test)
    return x - y

def test_func_call(x, y, **kwargs):
    return numba_func2(x, y, **kwargs)

def test_joe():
    test_func_call(1, 2, test="gabe")
