"""Task 2 — Multi-Core Scaling vs. Amdahl's Law (extension of Task 1)."""
import os

# Prevent hidden numerical-library worker pools from changing worker counts.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"

from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np

STUDENT_ID = 230103161
ARRAY_SIZE = 20_000_000
PARALLEL_FRACTION = 0.95
MAX_CORES = os.cpu_count() or 1
WORKER_COUNTS = list(dict.fromkeys([1, 2, 4, MAX_CORES, MAX_CORES * 2]))


def amdahl_bound(workers):
    return 1.0 / ((1.0 - PARALLEL_FRACTION) + PARALLEL_FRACTION / workers)


def chunk_sum(chunk):
    # NumPy releases the GIL during the large float64 reduction.
    return float(np.sum(chunk, dtype=np.float64))


def measure(pool, array, workers):
    # Split into non-overlapping contiguous views: no copying or serialization.
    chunks = np.array_split(array, workers)
    start = time.perf_counter()
    parts = list(pool.map(chunk_sum, chunks))
    total = sum(parts)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return total, elapsed_ms


def main():
    data = np.random.default_rng(STUDENT_ID).random(ARRAY_SIZE, dtype=np.float64)
    lines = [
        f"Student ID: {STUDENT_ID}",
        f"Array size: {ARRAY_SIZE:,} float64 elements",
        f"Logical cores (os.cpu_count): {MAX_CORES}",
        f"Parallelizable fraction (Amdahl P): {PARALLEL_FRACTION:.2f}",
        "Execution: ThreadPoolExecutor; NumPy chunk sums release the GIL",
        "Timing: wall-clock for dispatch, reduction, and collection; pool creation excluded",
        "",
        f"{'Cores':>7} | {'Wall Time (ms)':>15} | {'Speedup':>9} | {'Amdahl Upper Bound':>18}",
        "-" * 62,
    ]
    baseline = None
    # Create each pool outside the timed section, so setup costs aren't measured.
    for workers in WORKER_COUNTS:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            result, ms = measure(pool, data, workers)
        if baseline is None:
            baseline = ms
        # All runs must produce the same mathematical sum up to FP rounding.
        assert np.isclose(result, np.sum(data), rtol=1e-12, atol=1e-6), f"Wrong sum with {workers} workers"
        lines.append(f"{workers:7d} | {ms:15.3f} | {baseline / ms:8.3f}x | {amdahl_bound(workers):17.3f}x")
    lines.append("Verification: PASS")
    output = "\n".join(lines) + "\n"
    print(output, end="")
    with open("task2_output.txt", "w", encoding="utf-8") as file:
        file.write(output)


if __name__ == "__main__":
    main()
