"""Task 6: Synchronization Cost & Data Loss (Student ID: 230103161).

Uses NumPy's scalar storage and explicitly yields between read and write in the
unsafe case. This makes a lost-update race demonstrable on CPython builds
where a plain `counter += 1` might otherwise appear serialized by the GIL.
"""
import os
import sys
import time
import threading

STUDENT_ID = 230103161
THREADS = 4
INCREMENTS = 2_000_000
EXPECTED = THREADS * INCREMENTS


def benchmark(use_lock: bool):
    # A mutable shared counter, explicitly read/modify/write.
    counter = [0]
    lock = threading.Lock()
    barrier = threading.Barrier(THREADS + 1)

    def worker():
        barrier.wait()
        for i in range(INCREMENTS):
            if use_lock:
                with lock:
                    counter[0] += 1
            else:
                value = counter[0]
                # Cooperative switch explicitly exposes the unsafe read/write
                # interleaving, without adding sleep overhead on every update.
                if i % 10_000 == 0:
                    time.sleep(0)
                counter[0] = value + 1

    threads = [threading.Thread(target=worker) for _ in range(THREADS)]
    for thread in threads:
        thread.start()
    begin = time.perf_counter()
    barrier.wait()
    for thread in threads:
        thread.join()
    ms = (time.perf_counter() - begin) * 1000
    return counter[0], ms


def main():
    unsafe_result, unsafe_ms = benchmark(False)
    safe_result, safe_ms = benchmark(True)
    lost = EXPECTED - unsafe_result
    lost_percent = lost / EXPECTED * 100
    penalty_ms = safe_ms - unsafe_ms
    ratio = safe_ms / unsafe_ms if unsafe_ms else float('inf')
    output = (
        f"Student ID: {STUDENT_ID}\n"
        f"OS: {sys.platform}\n"
        f"Logical Cores: {os.cpu_count()}\n"
        f"Threads: {THREADS}\n"
        f"Increments per thread: {INCREMENTS:,}\n"
        f"Expected final value: {EXPECTED:,}\n"
        f"\nUNSAFE (no lock):\n"
        f"Actual final value: {unsafe_result:,}\n"
        f"Lost updates: {lost:,}\n"
        f"Lost updates percentage: {lost_percent:.4f}%\n"
        f"Wall time: {unsafe_ms:.3f} ms\n"
        f"\nSAFE (threading.Lock):\n"
        f"Actual final value: {safe_result:,}\n"
        f"Wall time: {safe_ms:.3f} ms\n"
        f"Synchronization time penalty: {penalty_ms:+.3f} ms\n"
        f"Time ratio (safe / unsafe): {ratio:.3f}x\n"
        f"Verification: {'PASS' if safe_result == EXPECTED else 'FAIL'}\n"
    )
    print(output, end='')
    with open('task6_output.txt', 'w', encoding='utf-8') as file:
        file.write(output)
    if safe_result != EXPECTED:
        sys.exit(1)


if __name__ == '__main__':
    main()
