"""Task 3: serial vs. parallel inclusive prefix scan.
Student ID: 230103161
Requires: numpy, numba (pip install numpy numba)
"""

import sys
from time import perf_counter

import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads

STUDENT_ID = 230103161
SIZE = 2_000_000
THREADS = 4


@njit
def serial_scan(values):
    result = np.empty(values.size, dtype=np.int64)
    running = np.int64(0)
    for i in range(values.size):
        running += values[i]
        result[i] = running
    return result


@njit(parallel=True)
def parallel_scan(values, chunks):
    """Two passes: local prefix scans, chunk offsets, then offset application."""
    n = values.size
    result = np.empty(n, dtype=np.int64)
    chunk_totals = np.zeros(chunks, dtype=np.int64)
    chunk_size = (n + chunks - 1) // chunks

    # Pass 1: independent inclusive scans within each chunk.
    for c in prange(chunks):
        start = c * chunk_size
        stop = min(start + chunk_size, n)
        total = np.int64(0)
        for i in range(start, stop):
            total += values[i]
            result[i] = total
        chunk_totals[c] = total

    # Offset reduction: exclusive prefix sum of chunk totals.
    offsets = np.empty(chunks, dtype=np.int64)
    accumulated = np.int64(0)
    for c in range(chunks):
        offsets[c] = accumulated
        accumulated += chunk_totals[c]

    # Pass 2: apply each chunk's offset in parallel.
    for c in prange(chunks):
        start = c * chunk_size
        stop = min(start + chunk_size, n)
        for i in range(start, stop):
            result[i] += offsets[c]

    return result


def measure(function, *args, repeats=5):
    timings = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = function(*args)
        timings.append((perf_counter() - start) * 1000)
    return result, min(timings)


def main():
    set_num_threads(min(THREADS, get_num_threads()))
    workers = get_num_threads()
    rng = np.random.default_rng(STUDENT_ID)
    values = rng.integers(1, 11, size=SIZE, dtype=np.int64)

    # Compile JIT kernels before measuring; avoid compilation overhead.
    sample = values[:1000]
    serial_scan(sample)
    parallel_scan(sample, workers)

    serial_result, serial_ms = measure(serial_scan, values)
    parallel_result, parallel_ms = measure(parallel_scan, values, workers)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Array Size: {SIZE:,}")
    print(f"Parallel Threads: {workers}")
    print(f"Serial Time (ms): {serial_ms:.3f}")
    print(f"Parallel Time (ms): {parallel_ms:.3f}")
    print(f"Speedup: {serial_ms / parallel_ms:.2f}x")

    try:
        assert parallel_result[-1] == serial_result[-1], "Last prefix sums differ"
        # Extra check for correctness of every prefix, beyond the requirement.
        assert np.array_equal(parallel_result, serial_result), "Prefix arrays differ"
    except AssertionError as error:
        print(f"Verification Status: FAIL ({error})")
        sys.exit(1)
    print(f"Final Prefix Sum: {int(parallel_result[-1])}")
    print("Verification Status: PASS")


if __name__ == "__main__":
    main()
