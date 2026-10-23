"""Task 1: parallel cache traversal benchmark (NumPy + Numba)."""

import os
import time
from pathlib import Path

import numpy as np
import psutil
from numba import njit, prange, set_num_threads, get_num_threads

STUDENT_ID = 230103161
N = 8000
OUTPUT_FILE = Path(__file__).with_name('task1_output.txt')


@njit(parallel=True)
def sum_row_major(a):
    """Workers own rows; the inner loop reads adjacent columns."""
    partial = np.zeros(a.shape[0], dtype=np.float64)
    for i in prange(a.shape[0]):
        s = 0.0
        for j in range(a.shape[1]):
            s += a[i, j]
        partial[i] = s
    return np.sum(partial)


@njit(parallel=True)
def sum_column_major(a):
    """Workers own columns; the inner loop jumps between rows."""
    partial = np.zeros(a.shape[1], dtype=np.float64)
    for j in prange(a.shape[1]):
        s = 0.0
        for i in range(a.shape[0]):
            s += a[i, j]
        partial[j] = s
    return np.sum(partial)


def measure(function, a):
    start = time.perf_counter()
    result = function(a)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def main():
    logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    physical = psutil.cpu_count(logical=False) or logical
    # Numba's configured maximum may be lower than the machine's physical count
    # in containers or restricted environments.
    from numba import config
    threads = min(physical, config.NUMBA_NUM_THREADS)
    set_num_threads(threads)

    rng = np.random.default_rng(STUDENT_ID)
    matrix = rng.random((N, N), dtype=np.float64)  # C order: rows contiguous

    # Compile once so JIT compilation time is excluded from timing.
    small = np.ones((8, 8), dtype=np.float64)
    sum_row_major(small)
    sum_column_major(small)

    row_sum, row_ms = measure(sum_row_major, matrix)
    col_sum, col_ms = measure(sum_column_major, matrix)
    if not np.isclose(row_sum, col_sum, rtol=1e-10, atol=1e-6):
        raise RuntimeError(f'Sums differ: {row_sum} vs {col_sum}')

    result = (
        f'Student ID: {STUDENT_ID}\n'
        f'Logical Cores: {logical}\n'
        f'Physical Cores: {physical}\n'
        f'Numba Threads Used: {get_num_threads()}\n'
        f'Matrix: {N} x {N}, float64\n'
        f'Mode A (Row-Major): {row_ms:.2f} ms\n'
        f'Mode B (Column-Major): {col_ms:.2f} ms\n'
        f'Cache Penalty Factor (B / A): {col_ms / row_ms:.2f}x\n'
        f'Mode A Sum: {row_sum:.6f}\n'
        f'Mode B Sum: {col_sum:.6f}\n'
    )
    print(result, end='')
    OUTPUT_FILE.write_text(result, encoding='utf-8')


if __name__ == '__main__':
    main()
