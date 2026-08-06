"""The decisive control: same sqlite reads, threads versus processes.

Everything so far says the collapse is not sqlite's doing — it survives
separate connections, separate database files (FanoutCache), a non-WAL
journal, a 256 MB page cache and a 1 GB mmap, and it does not respond to
`sys.setswitchinterval`. A plain dict, meanwhile, scales flat to 16 threads.

The one thing sqlite3 does that a dict does not is release the GIL around
every query. If that is the cause, the same work spread over processes —
which have no shared GIL — will scale. If processes collapse too, the cause
is the machine or the filesystem and the GIL is exonerated.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import queue
import random
import statistics
import sys
import tempfile
import threading
import time
import typing

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import common as c

WORKERS = [1, 2, 4, 8]
OPS = 20_000
TRIALS = 3
N_KEYS = 5_000
DIM = 1536


def _read_chunk(directory: str, seed: int, n_keys: int, ops: int, ready=None, elapsed=None) -> None:
    """Time the reads from inside the worker.

    A spawned process pays ~400 ms of interpreter startup. Measuring from the
    parent would fold that into the result and flatter the process numbers into
    meaninglessness, so every worker syncs on a barrier *after* it is warm and
    reports only its own read loop.
    """
    import diskcache

    cache = diskcache.Cache(directory=directory)
    keys = [f"k:{i}" for i in range(n_keys)]
    rng = random.Random(seed)
    cache.get(keys[0])  # warm the connection before the barrier
    if ready is not None:
        ready.wait()
    t0 = time.perf_counter()
    for _ in range(ops):
        cache.get(keys[rng.randrange(n_keys)])
    span = time.perf_counter() - t0
    cache.close()
    if elapsed is not None:
        elapsed.put(span)


def _drain(queue, n: int) -> float:
    """Aggregate ops/sec from per-worker spans: slowest worker sets the wall."""
    return max(queue.get() for _ in range(n))


def thread_run(directory: str, n: int, ops_each: int) -> float:
    barrier = threading.Barrier(n)
    q: typing.Any = queue.Queue()
    threads = [
        threading.Thread(target=_read_chunk, args=(directory, i, N_KEYS, ops_each, barrier, q)) for i in range(n)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return n * ops_each / _drain(q, n)


def process_run(directory: str, n: int, ops_each: int) -> float:
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(n)
    q = ctx.Queue()
    procs = [ctx.Process(target=_read_chunk, args=(directory, i, N_KEYS, ops_each, barrier, q)) for i in range(n)]
    for p in procs:
        p.start()
    spans = [q.get() for _ in range(n)]
    for p in procs:
        p.join()
    return n * ops_each / max(spans)


def sweep(label: str, runner, directory: str) -> list[dict]:
    rows, base = [], None
    for n in WORKERS:
        per = OPS // n
        trials = [runner(directory, n, per) for _ in range(TRIALS)]
        ops = statistics.median(trials)
        base = base if base is not None else ops
        rows.append({"workers": n, "ops_per_sec": ops, "scaling": ops / base})
    print(f"\n--- {label} ---")
    print(c.table(rows, ["workers", "ops_per_sec", "scaling"]))
    return rows


def startup_cost() -> float:
    """A spawned process pays interpreter startup; measure it so the process
    numbers can be read honestly rather than as if it were free."""
    ctx = mp.get_context("spawn")
    samples = []
    for _ in range(3):
        t0 = time.perf_counter()
        p = ctx.Process(target=_noop)
        p.start()
        p.join()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _noop() -> None:
    return None


def main() -> None:
    payload: dict = {"workers": WORKERS, "ops_total": OPS, "trials": TRIALS, "keys": N_KEYS}

    with tempfile.TemporaryDirectory() as d:
        cache = c.fresh_cache(d)
        c.warm_cache(cache, [f"k:{i}" for i in range(N_KEYS)], c.b64_vector(DIM))
        cache.close()

        payload["spawn_startup_seconds"] = startup_cost()
        print(f"spawn startup cost: {payload['spawn_startup_seconds'] * 1000:.0f} ms per process")

        payload["threads"] = sweep("threads (shared interpreter, shared GIL)", thread_run, d)
        payload["processes"] = sweep("processes (spawn, no shared GIL)", process_run, d)

    t8 = next(r for r in payload["threads"] if r["workers"] == 8)
    p8 = next(r for r in payload["processes"] if r["workers"] == 8)
    payload["process_over_thread_at_8"] = p8["ops_per_sec"] / t8["ops_per_sec"]
    print(f"\nAt 8 workers, processes are {payload['process_over_thread_at_8']:.1f}x the threaded throughput.")
    print(
        "Interpreter startup (~%.0f ms/process) is excluded: each worker times its own read loop\n"
        "after a barrier, and the slowest worker's span sets the wall." % (payload["spawn_startup_seconds"] * 1000)
    )

    out = c.RESULTS / "diag_proc_vs_thread.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
