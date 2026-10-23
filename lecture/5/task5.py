"""Task 5: serial/parallel overhead crossover, Student ID 230103161.
Requires: pip install numpy numba
Run: python task5.py
"""
import os
import platform
import time

import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads

STUDENT_ID=230103161
SIZES=[100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
A,B=2.5,1.75
REPEATS = 7


@njit(cache=True)
def serial_axb(x, y, a, b):
    for i in range(x.size):
        y[i] = a * x[i] + b


@njit(parallel=True, cache=True)
def parallel_axb(x, y, a, b):
    for i in prange(x.size):
        y[i] = a * x[i] + b


def measure(fn, x, y):
    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter_ns()
        fn(x, y, A, B)
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return float(np.median(samples))


def main():
    cores = os.cpu_count() or 1
    # Numba may have a smaller thread limit than os.cpu_count (e.g. CPU affinity).
    threads = get_num_threads()
    set_num_threads(threads)
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Logical cores (os.cpu_count): {cores}")
    print(f"Parallel worker threads: {threads}")
    print(f"Operation: y[i] = {A} * x[i] + {B}; median of {REPEATS} timed runs")
    print(f"{'N':>12} | {'Serial (ms)':>12} | {'Parallel (ms)':>13} | {'Speedup':>9} | {'Faster':>8}")
    print('-' * 69)

    # Compile and initialize the threading runtime before timing.
    warm_x = np.ones(1024, dtype=np.float64)
    warm_y = np.empty_like(warm_x)
    serial_axb(warm_x, warm_y, A, B)
    parallel_axb(warm_x, warm_y, A, B)

    results = []
    rng = np.random.default_rng(STUDENT_ID)
    for n in SIZES:
        x = rng.random(n, dtype=np.float64)
        ys = np.empty_like(x)
        yp = np.empty_like(x)
        serial_axb(x, ys, A, B)
        parallel_axb(x, yp, A, B)
        assert np.array_equal(ys, yp), f"Verification failed for N={n}"
        ts = measure(serial_axb, x, ys)
        tp = measure(parallel_axb, x, yp)
        label = 'Parallel' if tp < ts else ('Serial' if tp > ts else 'Tie')
        print(f"{n:12,d} | {ts:12.6f} | {tp:13.6f} | {ts/tp:8.3f}x | {label:>8}")
        results.append((n, ts, tp))

    crossovers = [(results[i-1][0], results[i][0]) for i in range(1, len(results))
                  if results[i-1][2] >= results[i-1][1] and results[i][2] < results[i][1]]
    if crossovers:
        print('Observed serial-to-parallel crossover interval(s): ' +
              ', '.join(f'{lo:,} < N <= {hi:,}' for lo, hi in crossovers))
    else:
        print('Observed serial-to-parallel crossover: NONE in tested sizes')
    print('NOTE: A crossover interval is bracketed by tested N values, not an exact single N.')
    print('Verification: PASS')


if __name__ == '__main__':
    main()
