"""Task 1 — Hardware Probe & Serial Baseline."""
import os

# Set single-threaded numerical-library limits before importing NumPy.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"

import platform
import time
import numpy as np

STUDENT_ID = 230103161
N = 20_000_000


def cpu_name():
    name = platform.processor().strip()
    if name:
        return name
    try:
        with open('/proc/cpuinfo', encoding='utf-8') as info:
            for line in info:
                if line.lower().startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.machine() or 'Unknown CPU'


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"CPU: {cpu_name()}")
    print(f"Logical cores (os.cpu_count): {os.cpu_count()}")
    print(f"Array size: {N:,} float64 elements")

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(N, dtype=np.float64)

    # NumPy reductions are serial for this 1D array; BLAS/thread backends
    # are also capped to one thread using environment variables above.
    start = time.perf_counter()
    total = float(np.sum(data, dtype=np.float64))
    std = float(np.std(data, dtype=np.float64, ddof=0))
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print(f"Sum: {total:.10f}")
    print(f"Standard deviation (population): {std:.10f}")
    print(f"Serial wall-clock time: {elapsed_ms:.3f} ms")

    with open('task1_output.txt', 'w', encoding='utf-8') as output:
        # Use the exact same reported values, but avoid a separate run changing timings.
        output.write(f"Student ID: {STUDENT_ID}\n")
        output.write(f"OS: {platform.platform()}\n")
        output.write(f"CPU: {cpu_name()}\n")
        output.write(f"Logical cores (os.cpu_count): {os.cpu_count()}\n")
        output.write(f"Array size: {N:,} float64 elements\n")
        output.write(f"Sum: {total:.10f}\n")
        output.write(f"Standard deviation (population): {std:.10f}\n")
        output.write(f"Serial wall-clock time: {elapsed_ms:.3f} ms\n")


if __name__ == '__main__':
    main()
