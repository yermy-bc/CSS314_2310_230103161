"""Task 4: IPC serialization vs zero-copy shared memory (CPU).
Student ID: 230103161. Requires Python 3.8+ and NumPy.
"""
import multiprocessing as mp
from multiprocessing import shared_memory
from time import perf_counter
import numpy as np

STUDENT_ID = 230103161
SIZES = (100_000, 1_000_000, 10_000_000)


def queue_worker(queue, result, ready):
    ready.put(True)
    data = queue.get()
    result.put((len(data), int(data[0]), int(data[-1])))


def shared_worker(name, shape, dtype, start, result, ready, go):
    shm = shared_memory.SharedMemory(name=name)
    try:
        arr = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)
        ready.put(True)
        go.wait()
        result.put((len(arr), int(arr[0]), int(arr[-1])))
    finally:
        shm.close()


def bench_queue(data, ctx):
    queue = ctx.Queue()
    result = ctx.Queue()
    ready = ctx.Queue()
    proc = ctx.Process(target=queue_worker, args=(queue, result, ready))
    proc.start()
    ready.get()
    t0 = perf_counter()
    queue.put(data)
    received = result.get()  # includes actual deserialization, not only queue.put
    elapsed = (perf_counter() - t0) * 1000
    proc.join()
    queue.close(); result.close(); ready.close()
    if proc.exitcode != 0:
        raise RuntimeError(f'Queue worker failed: {proc.exitcode}')
    return elapsed, received


def bench_shared(data, ctx):
    result = ctx.Queue()
    ready = ctx.Queue()
    go = ctx.Event()
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    try:
        proc = ctx.Process(target=shared_worker, args=(shm.name, data.shape, data.dtype.str, 0, result, ready, go))
        proc.start()
        ready.get()
        t0 = perf_counter()
        np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)[:] = data
        go.set()
        received = result.get()  # worker directly views the same memory
        elapsed = (perf_counter() - t0) * 1000
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f'Shared worker failed: {proc.exitcode}')
        return elapsed, received
    finally:
        shm.close()
        shm.unlink()
        result.close()
        ready.close()


def main():
    ctx = mp.get_context('spawn')
    rng = np.random.default_rng(STUDENT_ID)
    print(f'Student ID: {STUDENT_ID}')
    print('CPU IPC vs Shared Memory (int64 arrays)')
    print('Timing: sender transfer + receiver acknowledgement; worker initialization excluded from timer.')
    print(f'{"N":>12} | {"Path A Queue (ms)":>19} | {"Path B Shared (ms)":>19} | {"A/B":>7} | Status')
    for n in SIZES:
        data = rng.integers(1, 11, size=n, dtype=np.int64)
        expected = (n, int(data[0]), int(data[-1]))
        a, qa = bench_queue(data, ctx)
        b, sb = bench_shared(data, ctx)
        assert qa == expected and sb == expected, 'Transfer verification failed'
        print(f'{n:12,d} | {a:19.3f} | {b:19.3f} | {a/b:7.2f} | PASS', flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()
