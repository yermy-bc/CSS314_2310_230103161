"""Task 7: Arithmetic intensity versus memory-bandwidth scaling.

Requirements: pip install numpy numba
Run: python task7.py
Writes task7_output.txt using measurements from THIS computer.
"""
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads

STUDENT_ID = 230103161
N = 30_000_000
REPEATS = 3


@njit(parallel=True)
def vector_copy(a, b):
    for i in prange(a.size):
        b[i] = a[i]


@njit(parallel=True)
def trig_compute(a, b):
    for i in prange(a.size):
        x = a[i]
        b[i] = math.sin(x) * math.cos(x) + math.tan(x)


def benchmark(kernel, a, b, workers):
    set_num_threads(workers)
    measurements = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        kernel(a, b)
        measurements.append((time.perf_counter() - start) * 1000)
    return min(measurements)


def main():
    max_cores = min(os.cpu_count() or 1, __import__('numba').config.NUMBA_NUM_THREADS)
    configs = list(dict.fromkeys([1, 2 if max_cores >= 2 else 1, max_cores]))
    rng = np.random.default_rng(STUDENT_ID)
    # Keep values away from tangent singularities.
    a = rng.uniform(-0.75, 0.75, N).astype(np.float64)
    b = np.empty_like(a)

    # Compile both kernels OUTSIDE the timing window.
    set_num_threads(max_cores)
    small_a = a[:1000].copy()
    small_b = np.empty_like(small_a)
    vector_copy(small_a, small_b)
    trig_compute(small_a, small_b)

    results = {}
    for name, kernel in [('A - Vector copy', vector_copy),
                         ('B - Trigonometric', trig_compute)]:
        times = {p: benchmark(kernel, a, b, p) for p in configs}
        results[name] = times

    lines = [
        'TASK 7 - Arithmetic Intensity vs. Memory Bottlenecks',
        f'Student ID: {STUDENT_ID}',
        f'OS: {platform.platform()}',
        f'Logical cores detected: {os.cpu_count()}',
        f'Numba maximum threads: {max_cores}',
        f'Array length: {N:,} (float64)',
        f'Repeats per configuration: {REPEATS} (best elapsed time)',
        '',
    ]
    for name, times in results.items():
        lines.append(name)
        lines.append(f'{"Cores":>7} | {"Wall Time (ms)":>15} | {"Speedup":>9} | {"Efficiency":>11}')
        lines.append('-' * 54)
        baseline = times[1]
        for p in configs:
            speedup = baseline / times[p]
            lines.append(f'{p:>7} | {times[p]:>15.3f} | {speedup:>8.3f}x | {speedup/p:>10.3%}')
        lines.append('')

    if max_cores > 1:
        a_speed = results['A - Vector copy'][1] / results['A - Vector copy'][max_cores]
        b_speed = results['B - Trigonometric'][1] / results['B - Trigonometric'][max_cores]
        lines.append(f'At {max_cores} threads: copy speedup = {a_speed:.3f}x; compute speedup = {b_speed:.3f}x')
        if a_speed < b_speed:
            lines.append('Observed scaling stall: Kernel A (vector copy).')
            lines.append('Consistent with memory-bandwidth saturation; confirm using hardware counters.')
        elif a_speed > b_speed:
            lines.append('Observed scaling stall: Kernel B (trigonometric computation).')
            lines.append('Memory-bandwidth saturation of Kernel A is NOT established by these measurements.')
        else:
            lines.append('Observed scaling: both kernels show equal speedup; no unique stall identified.')
    else:
        lines.append('Only one core available; cannot identify a scaling stall.')
    lines.append('Note: run on your own CPU for machine-specific conclusions.')
    output = '\n'.join(lines) + '\n'
    print(output, end='')
    Path(__file__).with_name('task7_output.txt').write_text(output, encoding='utf-8')


if __name__ == '__main__':
    main()
