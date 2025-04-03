import timeit
from joblib import Parallel, delayed

def single_threaded_task(n):
    return sum(i * i for i in range(n))

def multi_threaded_task(n, n_jobs=4):
    return Parallel(n_jobs=n_jobs)(delayed(single_threaded_task)(n//n_jobs) for _ in range(n_jobs))

# Benchmarking
n = 10**6
single_time = timeit.timeit(lambda: single_threaded_task(n), number=5)
multi_time = timeit.timeit(lambda: multi_threaded_task(n), number=5)

print(f"Single-threaded execution time: {single_time:.4f} seconds")
print(f"Multi-threaded execution time: {multi_time:.4f} seconds")
