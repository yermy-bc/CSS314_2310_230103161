"""Task 8: CPU parallel matrix multiplication geometry benchmark.
Student ID: 230103161. Requires numpy and numba.
"""
import os
import platform
import time
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads

STUDENT_ID = 230103161
SIZE = 1024


@njit(parallel=True)
def matmul_naive(a, b):
    n = a.shape[0]
    c = np.zeros((n, n), dtype=np.float64)
    for i in prange(n):
        for j in range(n):
            acc = 0.0
            for k in range(n):
                acc += a[i, k] * b[k, j]
            c[i, j] = acc
    return c


@njit(parallel=True)
def matmul_tiled(a, b, tile):
    n = a.shape[0]
    c = np.zeros((n, n), dtype=np.float64)
    tiles_per_side = (n + tile - 1) // tile
    # Each worker owns disjoint output tile rows; no write races.
    for tile_i in prange(tiles_per_side):
        i0 = tile_i * tile
        i1 = min(i0 + tile, n)
        for tile_j in range(tiles_per_side):
            j0 = tile_j * tile
            j1 = min(j0 + tile, n)
            for tile_k in range(tiles_per_side):
                k0 = tile_k * tile
                k1 = min(k0 + tile, n)
                for i in range(i0, i1):
                    for k in range(k0, k1):
                        aik = a[i, k]
                        for j in range(j0, j1):
                            c[i, j] += aik * b[k, j]
    return c


def benchmark(fn, repetitions=3):
    measurements = []
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = fn()
        measurements.append((time.perf_counter() - start) * 1000)
    return min(measurements), result


def main():
    # Keep large matrix deterministic and float64.
    rng = np.random.default_rng(STUDENT_ID)
    a = rng.random((SIZE, SIZE), dtype=np.float64)
    b = rng.random((SIZE, SIZE), dtype=np.float64)
    max_threads = min(os.cpu_count() or 1, 64)
    set_num_threads(max_threads)

    lines = []
    def log(message=''):
        print(message)
        lines.append(str(message))

    log('TASK 8 - CPU Parallel Tile Reduction / Matrix Multiplication')
    log(f'Student ID: {STUDENT_ID}')
    log(f'OS: {platform.system()} ({platform.machine()})')
    log(f'Logical cores detected: {os.cpu_count()}')
    log(f'Numba threads used: {get_num_threads()}')
    log(f'Matrix size: {SIZE} x {SIZE} (float64)')
    log('Timings: minimum of 3 warmed-up wall-clock measurements; compilation excluded.')
    log('Note: cache misses are not directly measured; runtime is a cache-performance proxy.')

    # Compile outside measured section; warm each specialization first.
    matmul_naive(a[:2, :2].copy(), b[:2, :2].copy())
    matmul_tiled(a[:2, :2].copy(), b[:2, :2].copy(), 16)
    matmul_tiled(a[:2, :2].copy(), b[:2, :2].copy(), 64)

    measurements = {}
    output = {}
    for name, fn in [
        ('Non-tiled', lambda: matmul_naive(a, b)),
        ('16x16 tiled', lambda: matmul_tiled(a, b, 16)),
        ('64x64 tiled', lambda: matmul_tiled(a, b, 64)),
    ]:
        elapsed, result = benchmark(fn)
        measurements[name] = elapsed
        output[name] = result
        log(f'{name:<16} {elapsed:>12.3f} ms')

    # Numerical ordering differs across kernels; compare using floating tolerance.
    ref = output['Non-tiled']
    verification = all(np.allclose(ref, output[k], atol=1e-8, rtol=1e-10)
                       for k in ['16x16 tiled', '64x64 tiled'])
    log(f'16x16 vs non-tiled: {measurements["Non-tiled"] / measurements["16x16 tiled"]:.3f}x')
    log(f'64x64 vs non-tiled: {measurements["Non-tiled"] / measurements["64x64 tiled"]:.3f}x')
    log(f'Fastest configuration: {min(measurements, key=measurements.get)}')
    log(f'Verification: {"PASS" if verification else "FAIL"}')
    Path(__file__).with_name('task8_output.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if not verification:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
