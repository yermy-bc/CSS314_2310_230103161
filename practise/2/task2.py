"""Task 2 — PRNG state bottleneck (four Python threads, 40M points)."""
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103161
THREADS = 4
TOTAL_POINTS = 40_000_000
CHUNK_SIZE = 25_000


def run_mode(shared: bool):
    """Count Monte Carlo hits; shared mode synchronizes access to one RNG."""
    shared_rng = np.random.default_rng(STUDENT_ID * 10)
    rng_lock = threading.Lock()

    def worker(thread_index: int) -> int:
        rng = (shared_rng if shared else
               np.random.default_rng(STUDENT_ID * 10 + thread_index))
        count = TOTAL_POINTS // THREADS + (thread_index < TOTAL_POINTS % THREADS)
        hits = 0
        while count:
            n = min(CHUNK_SIZE, count)
            if shared:
                # Both draws belong to the same locked shared-PRNG operation.
                with rng_lock:
                    points = rng.random((n, 2))
            else:
                points = rng.random((n, 2))
            hits += int(np.count_nonzero(
                points[:, 0] ** 2 + points[:, 1] ** 2 <= 1.0
            ))
            count -= n
        return hits

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        hits = sum(executor.map(worker, range(THREADS)))
    elapsed_ms = (time.perf_counter() - start) * 1000
    pi_estimate = 4.0 * hits / TOTAL_POINTS
    return pi_estimate, abs(pi_estimate - math.pi), elapsed_ms


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"Total samples: {TOTAL_POINTS:,}")
    print(f"Threads: {THREADS}")
    print(f"Chunk size: {CHUNK_SIZE:,}")
    print()
    shared_pi, shared_error, shared_ms = run_mode(shared=True)
    independent_pi, independent_error, independent_ms = run_mode(shared=False)
    print(f"Shared PRNG:       pi = {shared_pi:.9f}, absolute error = {shared_error:.9f}, wall time = {shared_ms:.3f} ms")
    print(f"Independent PRNG:  pi = {independent_pi:.9f}, absolute error = {independent_error:.9f}, wall time = {independent_ms:.3f} ms")
    print(f"Speedup (shared / independent): {shared_ms / independent_ms:.2f}x")
    print("Note: timings depend on CPU, NumPy, and thread scheduling.")


if __name__ == "__main__":
    main()
