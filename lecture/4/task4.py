"""Task 4 — Work Stealing & Scheduling Under Skew (Python standard library)."""
import queue
import threading
import time
from pathlib import Path

STUDENT_ID = 230103161
ITEMS = 1000
WORKERS = 4
DYNAMIC_CHUNK_SIZE = 5


def simulate_work(i: int) -> None:
    """Irregular work explicitly requested by the assignment."""
    time.sleep(0.0001 * (i % 20))


def benchmark(strategy: str):
    ready = threading.Barrier(WORKERS + 1)
    finished_at = [0.0] * WORKERS
    counts = [0] * WORKERS
    tasks = queue.Queue()
    if strategy == 'Dynamic':
        for start in range(0, ITEMS, DYNAMIC_CHUNK_SIZE):
            tasks.put((start, min(start + DYNAMIC_CHUNK_SIZE, ITEMS)))

    def worker(worker_id: int):
        ready.wait()
        if strategy == 'Static':
            first = (worker_id * ITEMS) // WORKERS
            end = ((worker_id + 1) * ITEMS) // WORKERS
            for i in range(first, end):
                simulate_work(i)
                counts[worker_id] += 1
        else:
            while True:
                try:
                    first, end = tasks.get_nowait()
                except queue.Empty:
                    break
                for i in range(first, end):
                    simulate_work(i)
                    counts[worker_id] += 1
                tasks.task_done()
        finished_at[worker_id] = time.perf_counter()

    threads = [threading.Thread(target=worker, args=(i,), name=f'worker-{i}') for i in range(WORKERS)]
    for t in threads:
        t.start()
    start = time.perf_counter()
    ready.wait()  # release all workers together
    for t in threads:
        t.join()
    end = time.perf_counter()
    assert sum(counts) == ITEMS, f'Processed {sum(counts)} of {ITEMS} items'
    return {
        'wall_ms': (end - start) * 1000,
        'first_ms': (min(finished_at) - start) * 1000,
        'last_ms': (max(finished_at) - start) * 1000,
        'delta_ms': (max(finished_at) - min(finished_at)) * 1000,
        'counts': counts,
    }


def main():
    static = benchmark('Static')
    dynamic = benchmark('Dynamic')
    lines = [
        'TASK 4 — Work Stealing & Scheduling Under Skew',
        f'Student ID: {STUDENT_ID}',
        f'Items: {ITEMS} | Workers: {WORKERS} | Dynamic chunk size: {DYNAMIC_CHUNK_SIZE}',
        'Workload: sleep(0.0001 * (i % 20))',
        '',
        f'{"Strategy":<22} {"Wall Time (ms)":>16} {"First Finish (ms)":>19} {"Last Finish (ms)":>18} {"Finish Delta (ms)":>19}',
        '-' * 100,
    ]
    for label, result in [('A: Static chunks', static), ('B: Dynamic queue', dynamic)]:
        lines.append(f'{label:<22} {result["wall_ms"]:>16.3f} {result["first_ms"]:>19.3f} {result["last_ms"]:>18.3f} {result["delta_ms"]:>19.3f}')
        lines.append(f'  Items per worker: {result["counts"]}')
    lines.extend(['', f'Speedup (Static / Dynamic): {static["wall_ms"] / dynamic["wall_ms"]:.3f}x', 'Verification: PASS (1,000 items processed by each strategy)'])
    output = '\n'.join(lines) + '\n'
    print(output, end='')
    Path(__file__).with_name('task4_output.txt').write_text(output, encoding='utf-8')


if __name__ == '__main__':
    main()
