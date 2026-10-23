"""Task 3: False sharing experiment (Python + Numba, no external C compiler).

Install: python -m pip install numpy numba psutil
Run:     python task3.py

Note: JIT optimization and CPU architecture affect the observed ratio.
"""
import os
import platform
import threading
import time
from pathlib import Path

import numpy as np
from numba import njit

STUDENT_ID = 230103161
ITERATIONS = 50_000_000
OUTPUT_FILE = Path(__file__).with_name('task3_output.txt')


@njit(nogil=True)
def increment_shared(counts, index, n):
    for _ in range(n):
        counts[index] += 1


@njit(nogil=True)
def increment_local(counts, index, n):
    local = np.int64(0)
    for _ in range(n):
        local += 1
    counts[index] = local


def physical_workers():
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
    except ImportError:
        physical = None
    available = os.cpu_count() or 1
    if hasattr(os, 'sched_getaffinity'):
        available = len(os.sched_getaffinity(0))
    return max(1, min(physical or available, available))


def benchmark(kernel, workers):
    counts = np.zeros(workers, dtype=np.int64)
    gate = threading.Barrier(workers + 1)
    exceptions = []

    def worker(i):
        try:
            gate.wait()
            kernel(counts, i, ITERATIONS)
        except Exception as exc:
            exceptions.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(workers)]
    for t in threads:
        t.start()
    gate.wait()
    start = time.perf_counter()
    for t in threads:
        t.join()
    duration_ms = (time.perf_counter() - start) * 1000
    if exceptions:
        raise RuntimeError(f'Worker error: {exceptions[0]}')
    if not np.all(counts == ITERATIONS):
        raise AssertionError('Counter verification failed')
    return duration_ms


def main():
    workers = physical_workers()
    # Compile both functions BEFORE starting the clock.
    warm = np.zeros(1, dtype=np.int64)
    increment_shared(warm, 0, 1)
    increment_local(warm, 0, 1)

    unpadded = benchmark(increment_shared, workers)
    local = benchmark(increment_local, workers)
    ratio = unpadded / local if local else float('inf')
    report = (
        f'Student ID: {STUDENT_ID}\n'
        f'OS: {platform.platform()}\n'
        f'Logical cores: {os.cpu_count()}\n'
        f'Physical workers: {workers}\n'
        f'Increments per thread: {ITERATIONS:,}\n'
        f'Unpadded Shared: {unpadded:.3f} ms\n'
        f'Padded/Local: {local:.3f} ms\n'
        f'Execution Ratio (Unpadded / Local): {ratio:.3f}x\n'
        'Verification: PASS\n'
        'Note: JIT optimizations may reduce the observable false-sharing effect.\n'
    )
    print(report, end='')
    OUTPUT_FILE.write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
